#!/usr/bin/env bash
# One-time setup for a machine joining the cluster, on Linux or macOS.
# Windows uses scripts/setup.ps1, which does the same steps.
#
#   scripts/setup.sh                    # Qwen/Qwen3-0.6B, the review model
#   scripts/setup.sh Qwen/Qwen3-1.7B    # pre-download a different model
#
# Creates the venv at $TORRENT_VENV (default ~/Coding/venvs/torrent-llm),
# installs the package with the PyTorch build that matches this machine (CUDA
# if an NVIDIA driver is present, CPU otherwise), generates the gRPC stubs,
# downloads the model so nothing is fetched on the day, and runs the offline
# test suite as a smoke test. Safe to re-run.

set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model="${1:-Qwen/Qwen3-0.6B}"
venv="${TORRENT_VENV:-$HOME/Coding/venvs/torrent-llm}"

if ! command -v uv >/dev/null; then
  echo "uv is needed. Install it with:"
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

cd "$repo"
[ -x "$venv/bin/python" ] || uv venv "$venv" --python 3.12
# --torch-backend auto picks the CUDA wheel when it finds an NVIDIA driver.
VIRTUAL_ENV="$venv" uv pip install -e ".[api,dev]" --torch-backend auto
"$venv/bin/python" -m torrent_llm.codegen
echo "downloading $model (skipped if already cached)"
"$venv/bin/python" -c "from huggingface_hub import snapshot_download; snapshot_download('$model')"
"$venv/bin/python" -m pytest -q

"$venv/bin/python" -c "
import torch
from torrent_llm.shard import resolve_device
print(f'\nready: torch {torch.__version__}, device auto -> {resolve_device(\"auto\")}')
"
echo "next: see docs/multi-machine.md"
