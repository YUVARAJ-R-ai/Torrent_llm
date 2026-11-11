# One-command installer for Torrent-LLM on Windows.
# Linux and macOS use scripts/install.sh, which does the same steps.
#
#   powershell -ExecutionPolicy Bypass -c "& ([scriptblock]::Create((irm https://raw.githubusercontent.com/YUVARAJ-R-ai/Torrent_llm/main/scripts/install.ps1)))"
#   powershell -ExecutionPolicy Bypass -c "& ([scriptblock]::Create((irm https://raw.githubusercontent.com/YUVARAJ-R-ai/Torrent_llm/main/scripts/install.ps1))) 'tllm://join/...'"
#   powershell -ExecutionPolicy Bypass -c "& ([scriptblock]::Create((irm https://raw.githubusercontent.com/YUVARAJ-R-ai/Torrent_llm/main/scripts/install.ps1))) create"
#   powershell -ExecutionPolicy Bypass -c "& ([scriptblock]::Create((irm https://raw.githubusercontent.com/YUVARAJ-R-ai/Torrent_llm/main/scripts/install.ps1))) public"

param([string]$Target = "")

$ErrorActionPreference = "Stop"

# PowerShell does not stop on a failing native command, so check each one.
function Invoke-Step {
    & $args[0] @($args | Select-Object -Skip 1)
    if ($LASTEXITCODE -ne 0) { throw "failed: $($args -join ' ')" }
}

# 1. Argument: optional first argument.
# "tllm://join/..." -> join that swarm; "public" -> public swarm;
# "create" -> create a swarm; nothing -> just install and print how to start.
# Anything else -> print a clear error and exit 1.
# Never echo the invite link back in full (it is a secret): print "invite link received" instead.
$targetArg = if ($Target) { $Target } elseif ($args.Count -gt 0) { $args[0] } else { "" }
if ($args.Count -gt 1 -or ($Target -and $args.Count -gt 0)) {
    [Console]::Error.WriteLine("Error: Expected at most one argument ('create', 'public', 'tllm://join/...', or none).")
    exit 1
}

$mode = ""
if ($targetArg -ne "") {
    if ($targetArg -like "tllm://join/*") {
        $mode = "join"
        Write-Host "invite link received"
    }
    elseif ($targetArg -eq "public") {
        $mode = "public"
    }
    elseif ($targetArg -eq "create") {
        $mode = "create"
    }
    else {
        [Console]::Error.WriteLine("Error: Unknown argument. Expected 'create', 'public', 'tllm://join/...', or no argument.")
        exit 1
    }
}

# 2. App directory: $env:TORRENT_LLM_APP or default (%LOCALAPPDATA%\TorrentLLM\app).
# Venv: $env:TORRENT_VENV or <app dir>/../venv.
$app = if ($env:TORRENT_LLM_APP) {
    $env:TORRENT_LLM_APP.TrimEnd('\', '/')
} elseif ($env:LOCALAPPDATA) {
    Join-Path $env:LOCALAPPDATA "TorrentLLM\app"
} else {
    Join-Path $HOME "AppData\Local\TorrentLLM\app"
}
$appParent = Split-Path -Parent $app
$venv = if ($env:TORRENT_VENV) {
    $env:TORRENT_VENV.TrimEnd('\', '/')
} else {
    Join-Path $appParent "venv"
}

# 4b. SAFETY (MARKER): before replacing the app directory, refuse (exit 1 with a sentence)
# unless it is missing, empty, or contains the marker file `.torrent-llm-app`;
# never delete $HOME, / or a directory without the marker.
$markerName = ".torrent-llm-app"

function Assert-TargetSafety {
    $fullApp = [System.IO.Path]::GetFullPath($app).TrimEnd('\', '/')
    $fullHome = [System.IO.Path]::GetFullPath($HOME).TrimEnd('\', '/')
    $root = [System.IO.Path]::GetPathRoot($fullApp).TrimEnd('\', '/')

    if ($fullApp -eq $fullHome -or $fullApp -eq $root -or $fullApp -eq "") {
        [Console]::Error.WriteLine("Refusing to install into root or home directory ($app).")
        exit 1
    }

    if (Test-Path -LiteralPath $app) {
        $item = Get-Item -LiteralPath $app -Force
        $targetVal = if ($item.PSObject.Properties['Target']) { $item.Target } else { $null }
        $resolvedTarget = if ($targetVal) {
            $t = if ($targetVal -is [array]) { $targetVal[0] } else { $targetVal }
            # A relative link target is relative to the link's own folder, not to this
            # process's current directory.
            $base = Split-Path -Parent ([System.IO.Path]::GetFullPath($app))
            [System.IO.Path]::GetFullPath((Join-Path $base $t)).TrimEnd('\', '/')
        } else {
            $null
        }
        if ($resolvedTarget -and ($resolvedTarget -eq $fullHome -or $resolvedTarget -eq $root)) {
            [Console]::Error.WriteLine("Refusing to install into root or home directory ($app).")
            exit 1
        }
        $markerPath = Join-Path $app $markerName
        if (-not (Test-Path -LiteralPath $markerPath -PathType Leaf)) {
            $items = Get-ChildItem -LiteralPath $app -Force
            if ($items.Count -gt 0) {
                [Console]::Error.WriteLine("Refusing to overwrite $app: directory is not empty and does not contain $markerName.")
                exit 1
            }
        }
    }
}

Assert-TargetSafety

# 3. Install uv if missing: ps1 -> irm https://astral.sh/uv/install.ps1 | iex
# then add $HOME\.local\bin to $env:Path for this run.
$localBin = Join-Path $HOME ".local\bin"
if ($env:Path -split ";" -notcontains $localBin) {
    $env:Path = "$localBin;$env:Path"
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    if ($env:TORRENT_LLM_DRY_RUN -eq "1") {
        Write-Host "DRY: irm https://astral.sh/uv/install.ps1 | iex"
    } else {
        Write-Host "uv is needed. Installing uv..."
        try {
            irm https://astral.sh/uv/install.ps1 | iex
        } catch {
            [Console]::Error.WriteLine("Failed to install uv. Please check your internet connection: $_")
            exit 1
        }
    }
    if ($env:Path -split ";" -notcontains $localBin) {
        $env:Path = "$localBin;$env:Path"
    }
}

# 4. Get the code WITHOUT git: download zip or copy from $env:TORRENT_LLM_SOURCE
$tempDir = Join-Path ([System.IO.Path]::GetTempPath()) ("tllm-install-" + [System.Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $tempDir -Force | Out-Null

try {
    if ($env:TORRENT_LLM_SOURCE) {
        if (-not (Test-Path -LiteralPath $env:TORRENT_LLM_SOURCE)) {
            [Console]::Error.WriteLine("TORRENT_LLM_SOURCE directory does not exist: $env:TORRENT_LLM_SOURCE")
            exit 1
        }
        $extractedTop = Join-Path $tempDir "code"
        New-Item -ItemType Directory -Path $extractedTop -Force | Out-Null
        Copy-Item -Path (Join-Path $env:TORRENT_LLM_SOURCE "*") -Destination $extractedTop -Recurse -Force
    } else {
        $ref = if ($env:TORRENT_LLM_REF) { $env:TORRENT_LLM_REF } else { "main" }
        $url = "https://codeload.github.com/YUVARAJ-R-ai/Torrent_llm/zip/refs/heads/$ref"
        $zipPath = Join-Path $tempDir "archive.zip"
        $extractDir = Join-Path $tempDir "extracted"
        try {
            Invoke-WebRequest -Uri $url -OutFile $zipPath -UseBasicParsing
        } catch {
            [Console]::Error.WriteLine("Failed to download Torrent-LLM from $url. Please check your internet connection: $_")
            exit 1
        }
        Expand-Archive -Path $zipPath -DestinationPath $extractDir -Force
        $childDirs = Get-ChildItem -LiteralPath $extractDir -Directory
        if ($childDirs.Count -eq 1) {
            $extractedTop = $childDirs[0].FullName
        } else {
            $extractedTop = $extractDir
        }
    }

    if (-not (Test-Path -LiteralPath $appParent)) {
        New-Item -ItemType Directory -Path $appParent -Force | Out-Null
    }

    # Re-verify safety immediately before swap
    Assert-TargetSafety

    # Windows: Copy-Item -Recurse into a new sibling folder, then rename (no Move-Item across volumes)
    $stageFolder = Join-Path $appParent ("app.new." + [System.Guid]::NewGuid().ToString("N"))
    $appLeaf = Split-Path -Leaf $app
    try {
        Copy-Item -Path $extractedTop -Destination $stageFolder -Recurse -Force
        New-Item -ItemType File -Path (Join-Path $stageFolder $markerName) -Force | Out-Null
    } catch {
        Remove-Item -LiteralPath $stageFolder -Recurse -Force -ErrorAction SilentlyContinue
        [Console]::Error.WriteLine("Could not copy the new version (is the disk full?). Nothing was changed.")
        throw
    }

    if (Test-Path -LiteralPath $app) {
        $backupLeaf = "app.old." + [System.Guid]::NewGuid().ToString("N")
        $backupPath = Join-Path $appParent $backupLeaf
        try {
            Rename-Item -LiteralPath $app -NewName $backupLeaf
        } catch {
            Remove-Item -LiteralPath $stageFolder -Recurse -Force -ErrorAction SilentlyContinue
            [Console]::Error.WriteLine("Could not move the old version aside. Nothing was changed.")
            throw
        }
        try {
            Rename-Item -LiteralPath $stageFolder -NewName $appLeaf
        } catch {
            Remove-Item -LiteralPath $stageFolder -Recurse -Force -ErrorAction SilentlyContinue
            try {
                Rename-Item -LiteralPath $backupPath -NewName $appLeaf
                [Console]::Error.WriteLine("Could not put the new version in place; the old one was restored.")
            } catch {
                [Console]::Error.WriteLine("Could not put the new version in place, and putting the old one back failed too.")
                [Console]::Error.WriteLine("Your previous install is safe in: $backupPath")
            }
            throw
        }
        Remove-Item -LiteralPath $backupPath -Recurse -Force
    } else {
        try {
            Rename-Item -LiteralPath $stageFolder -NewName $appLeaf
        } catch {
            Remove-Item -LiteralPath $stageFolder -Recurse -Force -ErrorAction SilentlyContinue
            [Console]::Error.WriteLine("Could not put the new version in place.")
            throw
        }
    }
}
finally {
    Remove-Item -LiteralPath $tempDir -Recurse -Force -ErrorAction SilentlyContinue
}

# 5. Run the existing setup script from the app dir with TORRENT_SKIP_TESTS=1 and TORRENT_VENV set
$setupScript = Join-Path $app "scripts\setup.ps1"
if ($env:TORRENT_LLM_DRY_RUN -eq "1") {
    Write-Host "DRY: TORRENT_SKIP_TESTS=1 TORRENT_VENV=$venv powershell -ExecutionPolicy Bypass -File $setupScript"
} else {
    $env:TORRENT_SKIP_TESTS = "1"
    $env:TORRENT_VENV = $venv
    Push-Location $app
    try {
        & powershell -ExecutionPolicy Bypass -File $setupScript
        if ($LASTEXITCODE -ne 0) {
            [Console]::Error.WriteLine("Setup failed. Check the errors above.")
            exit 1
        }
    } finally {
        Pop-Location
    }
}

# 6. Start: <venv>\Scripts\torrent-peer.exe with --pool <app>\configs\pool.yaml and the mode/link
$peerExe = Join-Path $venv "Scripts\torrent-peer.exe"
$poolConfig = Join-Path $app "configs\pool.yaml"

if ($mode -eq "") {
    Write-Host ""
    Write-Host "Torrent-LLM is installed and ready."
    Write-Host "Start a peer using one of these commands:"
    Write-Host "  $peerExe --pool $poolConfig create"
    Write-Host "  $peerExe --pool $poolConfig join <link>"
    Write-Host "  $peerExe --pool $poolConfig public"
    exit 0
}

if ($env:TORRENT_LLM_DRY_RUN -eq "1") {
    if ($mode -eq "join") {
        Write-Host "DRY: $peerExe --pool $poolConfig join <link>"
    } else {
        Write-Host "DRY: $peerExe --pool $poolConfig $mode"
    }
    exit 0
}

if ($mode -eq "join") {
    & $peerExe --pool $poolConfig join $targetArg
    if ($LASTEXITCODE -ne 0) {
        [Console]::Error.WriteLine("torrent-peer stopped with exit code $LASTEXITCODE.")
        exit $LASTEXITCODE
    }
} else {
    Invoke-Step $peerExe --pool $poolConfig $mode
}
