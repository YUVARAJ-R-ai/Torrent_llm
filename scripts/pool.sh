#!/usr/bin/env bash
# Start the whole pool demo with one command: a lab of simulated devices (each
# its own process with its own memory budget and network link) and the dashboard.
# Ctrl+C stops all of it.
#
#   scripts/pool.sh                       # five devices, the pool dashboard
#   scripts/pool.sh --start 4             # fewer devices at first
#   scripts/pool.sh --pool my-pool.yaml   # another pool
#
# Open http://localhost:3000/pool when it says ready. The lab's own API is on
# :8100 and each device's HTTP port is 9101 and up.
#
# Expects the Python package installed (see docs/setup.md). If `torrent-lab` is
# not on PATH, the venv at $TORRENT_VENV (default ~/Coding/venvs/torrent-llm) is
# activated first. Logs go to $TORRENT_POOL_LOGS (default /tmp/torrent-pool).

set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
logs="${TORRENT_POOL_LOGS:-/tmp/torrent-pool}"
web_port="${TORRENT_WEB_PORT:-3000}"
pool_file="configs/pool-lab.yaml"
lab_args=()

while [ $# -gt 0 ]; do
  case "$1" in
    --pool) pool_file="$2"; shift 2 ;;
    --start) lab_args+=(--start "$2"); shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

cd "$repo"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
mkdir -p "$logs"

if ! command -v torrent-lab >/dev/null; then
  venv="${TORRENT_VENV:-$HOME/Coding/venvs/torrent-llm}"
  # shellcheck disable=SC1091
  source "$venv/bin/activate"
fi

# A port that is already taken would fail somewhere confusing and late.
for port in 8100 "$web_port"; do
  if ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":$port\$"; then
    echo "port $port is already in use; stop whatever is using it first" >&2
    exit 1
  fi
done

if [ ! -d dashboard/node_modules ]; then
  echo "installing dashboard dependencies (first run on this checkout)"
  (cd dashboard && npm install --no-audit --no-fund)
fi

pids=()

# Children first, then the process itself: the dashboard is npx -> next ->
# next-server, and the lab owns one process per device.
kill_tree() {
  local child
  for child in $(pgrep -P "$1"); do kill_tree "$child"; done
  kill "$1" 2>/dev/null || true
}

cleanup() {
  trap - EXIT INT TERM
  echo
  echo "stopping the pool"
  for pid in "${pids[@]}"; do kill_tree "$pid"; done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "starting the lab  (device logs: $logs/<name>.log)"
torrent-lab --pool "$pool_file" --log-dir "$logs" "${lab_args[@]}" >"$logs/lab.log" 2>&1 &
lab_pid=$!
pids+=("$lab_pid")

# The lab prints "lab ready" once its first device answers, and the rest are
# then loading the model in parallel.
until grep -q "lab ready" "$logs/lab.log" 2>/dev/null; do
  if ! kill -0 "$lab_pid" 2>/dev/null; then
    echo "the lab exited during startup:"
    tail -n 20 "$logs/lab.log"
    exit 1
  fi
  sleep 1
done

echo "starting the dashboard on :$web_port  (log: $logs/web.log)"
(cd dashboard && exec npx next dev --port "$web_port" >"$logs/web.log" 2>&1) &
pids+=("$!")
until curl -sf -o /dev/null "http://127.0.0.1:$web_port/pool"; do sleep 1; done

# Every device loads its share in the background. Wait until each stage has
# its copies, so the first thing on screen is a working pool, not a loading one.
echo "waiting for the devices to load the model"
for _ in $(seq 120); do
  state="$(curl -sf http://127.0.0.1:9101/pool 2>/dev/null || true)"
  if echo "$state" | grep -q '"health":"healthy"'; then break; fi
  sleep 1
done

echo
echo "ready: open http://localhost:$web_port/pool"
echo "Ctrl+C to stop everything"
wait
