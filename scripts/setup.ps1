# One-time setup for a Windows machine joining the cluster.
# Linux and macOS use scripts/setup.sh, which does the same steps.
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Model Qwen/Qwen3-1.7B
#
# Creates the venv at $env:TORRENT_VENV (default ~\Coding\venvs\torrent-llm),
# installs the package with the PyTorch build that matches this machine (CUDA
# if an NVIDIA driver is present, CPU otherwise), generates the gRPC stubs,
# downloads the model so nothing is fetched on the day, and runs the offline
# test suite as a smoke test. Safe to re-run.

param([string]$Model = "Qwen/Qwen3-0.6B")

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$venv = if ($env:TORRENT_VENV) { $env:TORRENT_VENV } else { Join-Path $HOME "Coding\venvs\torrent-llm" }
$py = Join-Path $venv "Scripts\python.exe"

# PowerShell does not stop on a failing native command, so check each one.
function Invoke-Step {
    & $args[0] @($args | Select-Object -Skip 1)
    if ($LASTEXITCODE -ne 0) { throw "failed: $($args -join ' ')" }
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv is needed. Install it with:"
    Write-Host '  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"'
    Write-Host "then open a new terminal and re-run this script."
    exit 1
}

Push-Location $repo
try {
    if (-not (Test-Path $py)) { Invoke-Step uv venv $venv --python 3.12 }
    $env:VIRTUAL_ENV = $venv
    # --torch-backend auto picks the CUDA wheel when it finds an NVIDIA driver.
    Invoke-Step uv pip install -e ".[api,dev]" --torch-backend auto
    Invoke-Step $py -m torrent_llm.codegen
    Write-Host "downloading $Model (skipped if already cached)"
    Invoke-Step $py -c "from huggingface_hub import snapshot_download; snapshot_download('$Model')"
    Invoke-Step $py -m pytest -q
    Invoke-Step $py -c "import torch; from torrent_llm.shard import resolve_device; print(f'\nready: torch {torch.__version__}, device auto -> {resolve_device('auto')}')"
    Write-Host "next: see docs\multi-machine.md"
}
finally {
    Pop-Location
}
