#!/usr/bin/env bash
# Start the whole showcase with one command: both shard "devices", the HTTP
# API, and the dashboard. Ctrl+C stops all of it.
#
#   scripts/demo.sh                          # the two-device demo rig
#   scripts/demo.sh configs/local-2shard.yaml  # any other topology
#
# Expects the Python package installed (see docs/setup.md). If `torrent-shard`
# is not on PATH, the venv at $TORRENT_VENV (default ~/Coding/venvs/torrent-llm)
# is activated first. Logs go to $TORRENT_DEMO_LOGS (default /tmp/torrent-demo).

set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config="${1:-configs/demo-2device.yaml}"
logs="${TORRENT_DEMO_LOGS:-/tmp/torrent-demo}"
api_port="${TORRENT_API_PORT:-8000}"
web_port="${TORRENT_WEB_PORT:-3000}"

cd "$repo"
# Shard and API output goes to log files, which Python would block-buffer;
# the readiness check below needs the "serving on" line as soon as it prints.
export PYTHONUNBUFFERED=1
mkdir -p "$logs"

if ! command -v torrent-shard >/dev/null; then
  venv="${TORRENT_VENV:-$HOME/Coding/venvs/torrent-llm}"
  # shellcheck disable=SC1091
  source "$venv/bin/activate"
fi

if ! python -c "import torrent_llm.transport" 2>/dev/null; then
  echo "generating gRPC stubs (first run on this checkout)"
  python -m torrent_llm.codegen
fi

if [ ! -d dashboard/node_modules ]; then
  echo "installing dashboard dependencies (first run on this checkout)"
  (cd dashboard && npm install --no-audit --no-fund)
fi

# Every background process we start, so one Ctrl+C (or one failure) stops all.
pids=()

# Children first, then the process itself. The dashboard is npx -> next ->
# next-server, and killing only the top of that chain leaves the server
# holding the port.
kill_tree() {
  local child
  for child in $(pgrep -P "$1"); do kill_tree "$child"; done
  kill "$1" 2>/dev/null || true
}

cleanup() {
  trap - EXIT INT TERM
  echo
  echo "stopping the demo"
  for pid in "${pids[@]}"; do kill_tree "$pid"; done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

num_nodes="$(python -c "from torrent_llm.runner import TopologyConfig as T; print(len(T.from_file('$config').nodes))")"

for ((i = 0; i < num_nodes; i++)); do
  echo "starting shard $i  (log: $logs/shard-$i.log)"
  torrent-shard --config "$config" --index "$i" >"$logs/shard-$i.log" 2>&1 &
  pids+=("$!")
done

# A shard prints "serving on" once its weights are loaded and the port is up.
# Loading takes a while on CPU, so wait for that rather than guessing a sleep.
for ((i = 0; i < num_nodes; i++)); do
  until grep -q "serving on" "$logs/shard-$i.log" 2>/dev/null; do
    if ! kill -0 "${pids[$i]}" 2>/dev/null; then
      echo "shard $i exited during startup:"
      tail -n 20 "$logs/shard-$i.log"
      exit 1
    fi
    sleep 1
  done
  echo "shard $i is up"
done

echo "starting the API on :$api_port  (log: $logs/api.log)"
torrent-api --config "$config" --port "$api_port" \
  --cors-origin "http://localhost:$web_port" \
  --cors-origin "http://127.0.0.1:$web_port" \
  >"$logs/api.log" 2>&1 &
pids+=("$!")

echo "starting the dashboard on :$web_port  (log: $logs/web.log)"
(cd dashboard && NEXT_PUBLIC_API_BASE="http://127.0.0.1:$api_port" \
  npx next dev --port "$web_port" >"$logs/web.log" 2>&1) &
pids+=("$!")

until curl -sf "http://127.0.0.1:$api_port/health" >/dev/null; do sleep 1; done
until curl -sf -o /dev/null "http://127.0.0.1:$web_port"; do sleep 1; done

# The first request on a GPU pays for CUDA kernel setup, about a second. Spend
# it here so the first run in front of an audience is as fast as the rest.
echo "warming up the model"
curl -sf -X POST "http://127.0.0.1:$api_port/generate" \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hi","max_new_tokens":2,"chat":true}' >/dev/null || true

echo
echo "ready: open http://localhost:$web_port"
echo "Ctrl+C to stop everything"
wait
