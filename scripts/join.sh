#!/usr/bin/env bash
# Join the pool from this machine (Linux and macOS). Windows uses scripts\join.ps1.
#
#   scripts/join.sh                          # find the others automatically
#   scripts/join.sh --join 192.168.1.20:9000 # or point it at one of them
#   scripts/join.sh --name alice --memory-mb 3000
#
# The first time, it installs what is needed and downloads the model (about 3 GB in
# all, so use a good connection). After that it only checks the machine and starts.
# Anything you add is passed to torrent-peer; `torrent-peer --help` lists the options.

set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv="${TORRENT_VENV:-$HOME/Coding/venvs/torrent-llm}"
py="$venv/bin/python"
cd "$repo"

if [ ! -x "$py" ] || ! "$py" -c "import torrent_llm" 2>/dev/null; then
  echo "First time on this machine: setting up. This downloads PyTorch and the model."
  if ! command -v uv >/dev/null; then
    echo "uv (a Python installer) is needed first."
    if [ -t 0 ]; then
      read -r -p "Install it now? [Y/n] " answer
      case "${answer:-Y}" in
        [Yy]*) curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH" ;;
        *) echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1 ;;
      esac
    else
      echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
      exit 1
    fi
  fi
  TORRENT_SKIP_TESTS=1 scripts/setup.sh
fi

# The gRPC files are generated, not stored, so a fresh pull needs them once.
"$py" -c "import torrent_llm.transport.activation_pb2" 2>/dev/null || "$py" -m torrent_llm.codegen

echo
echo "Checking this machine..."
if ! "$venv/bin/torrent-peer" --check "$@"; then
  echo
  echo "Fix the problems above, then run scripts/join.sh again."
  exit 1
fi

echo
exec "$venv/bin/torrent-peer" "$@"
