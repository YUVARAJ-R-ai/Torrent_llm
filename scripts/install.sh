#!/bin/sh
# One-command installer for Torrent-LLM on Linux and macOS.
# Windows uses scripts/install.ps1, which does the same steps.
#
#   curl -LsSf https://raw.githubusercontent.com/YUVARAJ-R-ai/Torrent_llm/main/scripts/install.sh | sh
#   curl -LsSf https://raw.githubusercontent.com/YUVARAJ-R-ai/Torrent_llm/main/scripts/install.sh | sh -s -- 'tllm://join/...'
#   curl -LsSf https://raw.githubusercontent.com/YUVARAJ-R-ai/Torrent_llm/main/scripts/install.sh | sh -s -- create
#   curl -LsSf https://raw.githubusercontent.com/YUVARAJ-R-ai/Torrent_llm/main/scripts/install.sh | sh -s -- public

set -eu

# 1. Argument: optional first argument.
# "tllm://join/..." -> join that swarm; "public" -> public swarm;
# "create" -> create a swarm; nothing -> just install and print how to start.
# Anything else -> print a clear error and exit 1.
# Never echo the invite link back in full (it is a secret): print "invite link received" instead.
if [ "$#" -gt 1 ]; then
  echo "Error: Expected at most one argument ('create', 'public', 'tllm://join/...', or none)." >&2
  exit 1
fi

arg="${1:-}"
mode=""
if [ -n "$arg" ]; then
  case "$arg" in
    tllm://join/*)
      mode="join"
      echo "invite link received"
      ;;
    public)
      mode="public"
      ;;
    create)
      mode="create"
      ;;
    *)
      echo "Error: Unknown argument. Expected 'create', 'public', 'tllm://join/...', or no argument." >&2
      exit 1
      ;;
  esac
fi

# 2. App directory: $TORRENT_LLM_APP or default (~/.local/share/torrent-llm/app on Linux,
# ~/Library/Application Support/TorrentLLM/app on macOS). Venv: $TORRENT_VENV or <app dir>/../venv.
if [ -n "${TORRENT_LLM_APP:-}" ]; then
  app="$TORRENT_LLM_APP"
else
  os_type="$(uname -s 2>/dev/null || echo "Linux")"
  case "$os_type" in
    Darwin)
      app="$HOME/Library/Application Support/TorrentLLM/app"
      ;;
    *)
      app="$HOME/.local/share/torrent-llm/app"
      ;;
  esac
fi

app_display="${TORRENT_LLM_APP:-$app}"
app="${app%/}"
app_parent="$(dirname "$app")"
venv="${TORRENT_VENV:-$app_parent/venv}"
venv="${venv%/}"

# 4b. SAFETY (MARKER): before replacing the app directory, refuse (exit 1 with a sentence)
# unless it is missing, empty, or contains the marker file `.torrent-llm-app`;
# never delete $HOME, / or a directory without the marker.
is_dir_empty() {
  _chk_dir="$1"
  [ -d "$_chk_dir" ] || return 1
  for _item in "$_chk_dir"/* "$_chk_dir"/.[!.]* "$_chk_dir"/..?*; do
    if [ -e "$_item" ] || [ -L "$_item" ]; then
      return 1
    fi
  done
  return 0
}

check_safety() {
  _app_canon=""
  if [ -d "$app" ]; then
    _app_canon="$(cd "$app" 2>/dev/null && pwd -P || true)"
  fi
  _home_canon="$(cd "$HOME" 2>/dev/null && pwd -P || true)"

  if [ -z "$app" ] || [ "$app" = "/" ] || [ "$app" = "$HOME" ] || [ "$_app_canon" = "/" ] || { [ -n "$_home_canon" ] && [ "$_app_canon" = "$_home_canon" ]; }; then
    echo "Refusing to install into root or home directory (${TORRENT_LLM_APP:-$app_display})." >&2
    exit 1
  fi

  if [ -e "$app" ] || [ -L "$app" ]; then
    if [ ! -d "$app" ]; then
      echo "Refusing to overwrite $app: target path exists and is not a directory." >&2
      exit 1
    fi
    if [ -f "$app/.torrent-llm-app" ] && [ ! -L "$app/.torrent-llm-app" ]; then
      :
    else
      if ! is_dir_empty "$app"; then
        echo "Refusing to overwrite $app: directory is not empty and does not contain .torrent-llm-app." >&2
        exit 1
      fi
    fi
  fi
}

check_safety

# 3. Install uv if missing: sh -> curl -LsSf https://astral.sh/uv/install.sh | sh
# then add $HOME/.local/bin to PATH for this run.
case ":$PATH:" in
  *:"$HOME/.local/bin":*) ;;
  *) PATH="$HOME/.local/bin:$PATH"; export PATH ;;
esac

if ! command -v uv >/dev/null 2>&1; then
  if [ "${TORRENT_LLM_DRY_RUN:-0}" = "1" ]; then
    echo "DRY: curl -LsSf https://astral.sh/uv/install.sh | sh"
  else
    if ! command -v curl >/dev/null 2>&1; then
      echo "curl is needed to install uv; please install curl." >&2
      exit 1
    fi
    echo "uv is needed. Installing uv..."
    # Download first, then run: a pipe would hide a failed download (POSIX sh has no
    # pipefail, and sh happily runs an empty script).
    uv_script="$(mktemp "${TMPDIR:-/tmp}/uv-install.XXXXXX")"
    if ! curl -LsSf https://astral.sh/uv/install.sh -o "$uv_script" || ! sh "$uv_script"; then
      rm -f "$uv_script"
      echo "Failed to install uv. Please check your internet connection." >&2
      exit 1
    fi
    rm -f "$uv_script"
  fi
fi

# Pre-flight check for bash before any download or swap (setup.sh requires bash)
if [ "${TORRENT_LLM_DRY_RUN:-0}" != "1" ]; then
  if ! command -v bash >/dev/null 2>&1; then
    echo "bash is needed; install it" >&2
    exit 1
  fi
fi

# 4. Get the code WITHOUT git: download tar.gz or copy from $TORRENT_LLM_SOURCE
tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/tllm-install.XXXXXX" 2>/dev/null || mktemp -d 2>/dev/null || mktemp -d -t 'tllm-install.XXXXXX')"
stage=""
old_holder=""
cleanup() {
  rm -rf "$tmp_dir"
  # A half-made copy of the new version never stays behind.
  if [ -n "$stage" ]; then rm -rf "$stage"; fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP

extracted_top=""
if [ -n "${TORRENT_LLM_SOURCE:-}" ]; then
  if [ ! -d "$TORRENT_LLM_SOURCE" ]; then
    echo "TORRENT_LLM_SOURCE directory does not exist: $TORRENT_LLM_SOURCE" >&2
    exit 1
  fi
  extracted_top="$tmp_dir/code"
  mkdir -p "$extracted_top"
  src_clean="${TORRENT_LLM_SOURCE%/}"
  cp -R "$src_clean"/. "$extracted_top/"
else
  if ! command -v curl >/dev/null 2>&1; then
    echo "curl is needed to download Torrent-LLM; please install curl." >&2
    exit 1
  fi
  if ! command -v tar >/dev/null 2>&1; then
    echo "tar is needed to extract Torrent-LLM; please install tar." >&2
    exit 1
  fi
  ref="${TORRENT_LLM_REF:-main}"
  url="https://codeload.github.com/YUVARAJ-R-ai/Torrent_llm/tar.gz/refs/heads/$ref"
  if ! curl -LsSf "$url" -o "$tmp_dir/archive.tar.gz"; then
    echo "Failed to download Torrent-LLM from $url. Please check your internet connection." >&2
    exit 1
  fi
  mkdir -p "$tmp_dir/extracted"
  if ! tar -xzf "$tmp_dir/archive.tar.gz" -C "$tmp_dir/extracted"; then
    echo "Failed to extract Torrent-LLM archive." >&2
    exit 1
  fi
  for _d in "$tmp_dir/extracted"/*; do
    if [ -d "$_d" ]; then
      extracted_top="$_d"
      break
    fi
  done
  if [ -z "$extracted_top" ] || [ ! -d "$extracted_top" ]; then
    extracted_top="$tmp_dir/extracted"
  fi
fi

mkdir -p "$app_parent"

# Re-check safety immediately before swapping to avoid TOCTOU
check_safety

# Replace the app dir with the new version: copy beside it, move the old one aside,
# move the new one in, and only then delete the old one. Every failure says exactly
# where things are.
stage="$(mktemp -d "$app_parent/.tllm-stage.XXXXXX")"
if ! cp -R "$extracted_top"/. "$stage/"; then
  echo "Could not copy the new version (is the disk full?). Nothing was changed." >&2
  exit 1
fi
touch "$stage/.torrent-llm-app"

if [ -e "$app" ] || [ -L "$app" ]; then
  old_holder="$(mktemp -d "$app_parent/.tllm-old.XXXXXX")"
  if ! mv "$app" "$old_holder/app"; then
    rmdir "$old_holder" 2>/dev/null || true
    echo "Could not move the old version aside. Nothing was changed." >&2
    exit 1
  fi
fi

if ! mv "$stage" "$app"; then
  if [ -n "$old_holder" ]; then
    if mv "$old_holder/app" "$app"; then
      rmdir "$old_holder" 2>/dev/null || true
      echo "Could not put the new version in place; the old one was restored." >&2
    else
      echo "Could not put the new version in place, and putting the old one back failed too." >&2
      echo "Your previous install is safe in: $old_holder/app" >&2
    fi
  else
    echo "Could not put the new version in place." >&2
  fi
  exit 1
fi
stage=""  # it is the app now; nothing to clean up

if [ -n "$old_holder" ]; then
  rm -rf "$old_holder"
fi

# Clean up download/copy temporary directory immediately after swap
rm -rf "$tmp_dir"
trap - EXIT INT TERM HUP

# 5. Run the existing setup script from the app dir with TORRENT_SKIP_TESTS=1 and TORRENT_VENV set
cd "$app"
if [ "${TORRENT_LLM_DRY_RUN:-0}" = "1" ]; then
  echo "DRY: TORRENT_SKIP_TESTS=1 TORRENT_VENV=$venv bash scripts/setup.sh"
else
  export TORRENT_SKIP_TESTS=1
  export TORRENT_VENV="$venv"
  if ! bash scripts/setup.sh; then
    echo "Setup failed. Check the errors above." >&2
    exit 1
  fi
fi

# 6. Start: <venv>/bin/torrent-peer with --pool <app>/configs/pool.yaml and the mode/link
if [ -z "$mode" ]; then
  echo
  echo "Torrent-LLM is installed and ready."
  echo "Start a peer using one of these commands:"
  echo "  \"$venv/bin/torrent-peer\" --pool \"$app/configs/pool.yaml\" create"
  echo "  \"$venv/bin/torrent-peer\" --pool \"$app/configs/pool.yaml\" join <link>"
  echo "  \"$venv/bin/torrent-peer\" --pool \"$app/configs/pool.yaml\" public"
  exit 0
fi

if [ "${TORRENT_LLM_DRY_RUN:-0}" = "1" ]; then
  if [ "$mode" = "join" ]; then
    echo "DRY: \"$venv/bin/torrent-peer\" --pool \"$app/configs/pool.yaml\" join <link>"
  else
    echo "DRY: \"$venv/bin/torrent-peer\" --pool \"$app/configs/pool.yaml\" $mode"
  fi
  exit 0
fi

if [ "$mode" = "join" ]; then
  exec "$venv/bin/torrent-peer" --pool "$app/configs/pool.yaml" join "$arg"
else
  exec "$venv/bin/torrent-peer" --pool "$app/configs/pool.yaml" "$mode"
fi
