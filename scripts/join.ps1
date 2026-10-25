# Join the pool from this machine (Windows). Linux and macOS use scripts/join.sh.
#
#   powershell -ExecutionPolicy Bypass -File scripts\join.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\join.ps1 --join 192.168.1.20:9000
#   powershell -ExecutionPolicy Bypass -File scripts\join.ps1 --name alice --memory-mb 3000
#
# The first time, it installs what is needed and downloads the model (about 3 GB in
# all, so use a good connection). After that it only checks the machine and starts.
# Anything you add is passed to torrent-peer; `torrent-peer --help` lists the options.

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$venv = if ($env:TORRENT_VENV) { $env:TORRENT_VENV } else { Join-Path $HOME "Coding\venvs\torrent-llm" }
$py = Join-Path $venv "Scripts\python.exe"
$peer = Join-Path $venv "Scripts\torrent-peer.exe"
$extra = $args

Push-Location $repo
try {
    $ready = $false
    if (Test-Path $py) {
        & $py -c "import torrent_llm" 2>$null
        $ready = ($LASTEXITCODE -eq 0)
    }
    if (-not $ready) {
        Write-Host "First time on this machine: setting up. This downloads PyTorch and the model."
        if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
            Write-Host "uv (a Python installer) is needed first. Install it with:"
            Write-Host '  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"'
            Write-Host "then open a new terminal and run this script again."
            exit 1
        }
        $env:TORRENT_SKIP_TESTS = "1"
        & powershell -ExecutionPolicy Bypass -File (Join-Path $repo "scripts\setup.ps1")
        if ($LASTEXITCODE -ne 0) { throw "setup failed" }
    }

    # The gRPC files are generated, not stored, so a fresh pull needs them once.
    & $py -c "import torrent_llm.transport.activation_pb2" 2>$null
    if ($LASTEXITCODE -ne 0) { & $py -m torrent_llm.codegen }

    Write-Host ""
    Write-Host "Checking this machine..."
    & $peer --check @extra
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "Fix the problems above, then run scripts\join.ps1 again."
        exit 1
    }

    Write-Host ""
    & $peer @extra
}
finally {
    Pop-Location
}
