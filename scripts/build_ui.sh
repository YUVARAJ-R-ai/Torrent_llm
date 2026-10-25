#!/usr/bin/env bash
# Build the dashboard as static files and put them where every device serves them
# from, so a device needs only Python to show the pool in a browser.
#
#   scripts/build_ui.sh
#
# Needs Node (only for this step, and only for whoever changes the dashboard). The
# result is committed under src/torrent_llm/pool/ui, so everyone else never runs it.
# Run it again after changing anything under dashboard/.

set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target="$repo/src/torrent_llm/pool/ui"

cd "$repo/dashboard"
[ -d node_modules ] || npm install --no-audit --no-fund
rm -rf out
NEXT_EXPORT=1 NEXT_PUBLIC_POOL_SAME_ORIGIN=1 npm run build

rm -rf "$target"
cp -r out "$target"
echo "built the dashboard into $target ($(du -sh "$target" | cut -f1))"
