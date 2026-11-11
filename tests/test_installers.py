"""Tests for the Linux/macOS and Windows one-command swarm installers.

Verifies syntax validity under POSIX sh and dash, contract enforcement,
argument handling, invite link secrecy, the directory safety marker guard,
and static checks for the PowerShell installer.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"
INSTALL_PS1 = REPO_ROOT / "scripts" / "install.ps1"


def _make_fake_uv(bin_dir: Path) -> None:
    """Create a minimal executable fake uv binary script in bin_dir."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    uv_file = bin_dir / "uv"
    uv_file.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    uv_file.chmod(uv_file.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _make_fake_source(source_dir: Path) -> None:
    """Populate source_dir with scripts/setup.sh and configs/pool.yaml."""
    scripts_dir = source_dir / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    setup_file = scripts_dir / "setup.sh"
    setup_file.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    setup_file.chmod(setup_file.stat().st_mode | stat.S_IXUSR)

    configs_dir = source_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    pool_file = configs_dir / "pool.yaml"
    pool_file.write_text("model: test-model\n", encoding="utf-8")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell test")
def test_sh_syntax() -> None:
    """Verify scripts/install.sh passes POSIX sh syntax checking."""
    proc = subprocess.run(["sh", "-n", str(INSTALL_SH)], capture_output=True, text=True)
    assert proc.returncode == 0, f"sh -n failed: {proc.stderr}"
    assert proc.stdout == ""
    assert proc.stderr == ""


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell test")
def test_dash_syntax() -> None:
    """Verify scripts/install.sh passes dash syntax checking if dash is installed."""
    dash_bin = shutil.which("dash")
    if not dash_bin:
        pytest.skip("dash is not installed")
    proc = subprocess.run([dash_bin, "-n", str(INSTALL_SH)], capture_output=True, text=True)
    assert proc.returncode == 0, f"dash -n failed: {proc.stderr}"
    assert proc.stdout == ""
    assert proc.stderr == ""


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell test")
@pytest.mark.parametrize("shell_bin", ["sh", "dash"])
def test_dry_run_join_link(tmp_path: Path, shell_bin: str) -> None:
    """Verify dry run joins an invite link without echoing the link and prints DRY commands."""
    if shell_bin == "dash" and not shutil.which("dash"):
        pytest.skip("dash is not installed")

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    fake_bin = tmp_path / "bin"
    _make_fake_uv(fake_bin)

    source_dir = tmp_path / "source"
    _make_fake_source(source_dir)

    app_dir = tmp_path / "app"

    env = {
        **os.environ,
        "HOME": str(fake_home),
        "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}",
        "TORRENT_LLM_DRY_RUN": "1",
        "TORRENT_LLM_SOURCE": str(source_dir),
        "TORRENT_LLM_APP": str(app_dir),
    }

    link = "tllm://join/abc"
    proc = subprocess.run(
        [shell_bin, str(INSTALL_SH), link],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"installer failed: {proc.stderr}\nstdout: {proc.stdout}"
    output = proc.stdout + proc.stderr
    assert "invite link received" in output
    assert link not in output, "Invite link was echoed back, violating secrecy"

    dry_lines = [line for line in output.splitlines() if line.startswith("DRY: ")]
    assert any("torrent-peer" in line and "join" in line for line in dry_lines)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell test")
def test_unknown_argument_exits_one(tmp_path: Path) -> None:
    """Verify an unknown argument causes the installer to exit 1 without echoing secrets."""
    proc = subprocess.run(
        ["sh", str(INSTALL_SH), "tllm://joinX/secret"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1
    assert "Expected" in proc.stderr or "Unknown argument" in proc.stderr
    output = proc.stdout + proc.stderr
    assert "secret" not in output, "Bad argument value was echoed back in output"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell test")
def test_marker_guard_protects_non_empty_directory(tmp_path: Path) -> None:
    """Verify the installer refuses to overwrite a non-empty directory without .torrent-llm-app."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    fake_bin = tmp_path / "bin"
    _make_fake_uv(fake_bin)

    source_dir = tmp_path / "source"
    _make_fake_source(source_dir)

    app_dir = tmp_path / "app"
    app_dir.mkdir()
    secret_file = app_dir / "user_data.txt"
    secret_file.write_text("critical content", encoding="utf-8")

    env = {
        **os.environ,
        "HOME": str(fake_home),
        "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}",
        "TORRENT_LLM_DRY_RUN": "1",
        "TORRENT_LLM_SOURCE": str(source_dir),
        "TORRENT_LLM_APP": str(app_dir),
    }

    proc = subprocess.run(
        ["sh", str(INSTALL_SH)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1
    assert "Refusing to overwrite" in proc.stderr
    assert ".torrent-llm-app" in proc.stderr
    assert secret_file.exists()
    assert secret_file.read_text(encoding="utf-8") == "critical content"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell test")
def test_marker_guard_allows_replacement_when_marker_present(tmp_path: Path) -> None:
    """Verify the installer replaces the app directory when .torrent-llm-app is present."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    fake_bin = tmp_path / "bin"
    _make_fake_uv(fake_bin)

    source_dir = tmp_path / "source"
    _make_fake_source(source_dir)

    app_dir = tmp_path / "app"
    app_dir.mkdir()
    marker = app_dir / ".torrent-llm-app"
    marker.touch()
    old_file = app_dir / "old_version.txt"
    old_file.write_text("old", encoding="utf-8")

    env = {
        **os.environ,
        "HOME": str(fake_home),
        "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}",
        "TORRENT_LLM_DRY_RUN": "1",
        "TORRENT_LLM_SOURCE": str(source_dir),
        "TORRENT_LLM_APP": str(app_dir),
    }

    proc = subprocess.run(
        ["sh", str(INSTALL_SH)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"installer failed: {proc.stderr}\nstdout: {proc.stdout}"
    assert not old_file.exists()
    assert (app_dir / "configs" / "pool.yaml").exists()
    assert (app_dir / ".torrent-llm-app").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell test")
@pytest.mark.parametrize(
    "target_case",
    ["home_with_marker", "symlink_to_home", "root", "marker_as_directory"],
)
def test_marker_guard_refuses_unsafe_targets(tmp_path: Path, target_case: str) -> None:
    """Verify the marker guard refuses root, home, symlink to home, or directory marker."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    home_canary = fake_home / "canary.txt"
    home_canary.write_text("home_canary", encoding="utf-8")

    fake_bin = tmp_path / "bin"
    _make_fake_uv(fake_bin)

    source_dir = tmp_path / "source"
    _make_fake_source(source_dir)

    env = {
        **os.environ,
        "HOME": str(fake_home),
        "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}",
        "TORRENT_LLM_DRY_RUN": "1",
        "TORRENT_LLM_SOURCE": str(source_dir),
    }

    if target_case == "home_with_marker":
        (fake_home / ".torrent-llm-app").touch()
        env["TORRENT_LLM_APP"] = str(fake_home)
        proc = subprocess.run(["sh", str(INSTALL_SH)], env=env, capture_output=True, text=True)
        assert proc.returncode == 1
        assert "Refusing to install into root or home directory" in proc.stderr
        assert home_canary.read_text(encoding="utf-8") == "home_canary"

    elif target_case == "symlink_to_home":
        symlink_dir = tmp_path / "link_to_home"
        symlink_dir.symlink_to(fake_home)
        env["TORRENT_LLM_APP"] = str(symlink_dir)
        proc = subprocess.run(["sh", str(INSTALL_SH)], env=env, capture_output=True, text=True)
        assert proc.returncode == 1
        assert "Refusing to install into root or home directory" in proc.stderr
        assert home_canary.read_text(encoding="utf-8") == "home_canary"

    elif target_case == "root":
        env["TORRENT_LLM_APP"] = "/"
        proc = subprocess.run(["sh", str(INSTALL_SH)], env=env, capture_output=True, text=True)
        assert proc.returncode == 1
        assert "Refusing to install into root or home directory (/)" in proc.stderr

    elif target_case == "marker_as_directory":
        app_dir = tmp_path / "app"
        app_dir.mkdir()
        marker_dir = app_dir / ".torrent-llm-app"
        marker_dir.mkdir()
        app_canary = app_dir / "canary.txt"
        app_canary.write_text("marker_dir_canary", encoding="utf-8")
        env["TORRENT_LLM_APP"] = str(app_dir)
        proc = subprocess.run(["sh", str(INSTALL_SH)], env=env, capture_output=True, text=True)
        assert proc.returncode == 1
        assert "Refusing to overwrite" in proc.stderr
        assert app_canary.read_text(encoding="utf-8") == "marker_dir_canary"
        assert marker_dir.is_dir()


def test_install_ps1_static_checks() -> None:
    """Verify install.ps1 satisfies required static safety and implementation contracts."""
    content = INSTALL_PS1.read_text(encoding="utf-8")
    assert '$ErrorActionPreference = "Stop"' in content
    assert "Invoke-Step" in content
    assert "codeload.github.com" in content
    assert "TORRENT_LLM_DRY_RUN" in content
    assert "git clone" not in content
    assert not re.search(r"Invoke-Step.*targetArg", content), "targetArg passed to Invoke-Step"
    assert "-LiteralPath" in content
    assert "-PathType Leaf" in content
    assert "TORRENT_SKIP_TESTS=1 TORRENT_VENV=$venv powershell" in content


# --- manager regressions ------------------------------------------------------


def _dry_run(tmp_path, *args, extra_env=None):
    import os as _os
    import subprocess as _sp

    source = tmp_path / "src"
    (source / "scripts").mkdir(parents=True)
    (source / "configs").mkdir()
    (source / "scripts" / "setup.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
    (source / "configs" / "pool.yaml").write_text("model_id: x\n")
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    (fakebin / "uv").write_text("#!/bin/sh\nexit 0\n")
    (fakebin / "uv").chmod(0o755)
    env = {
        **_os.environ,
        "HOME": str(tmp_path / "home"),
        "TORRENT_LLM_APP": str(tmp_path / "home" / "app"),
        "TORRENT_LLM_SOURCE": str(source),
        "TORRENT_LLM_DRY_RUN": "1",
        "PATH": f"{fakebin}:{_os.environ['PATH']}",
        **(extra_env or {}),
    }
    (tmp_path / "home").mkdir(exist_ok=True)
    return _sp.run(["sh", "scripts/install.sh", *args], env=env, capture_output=True, text=True)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell")
def test_dry_run_runs_setup_with_bash_and_leaves_no_staging_behind(tmp_path):
    result = _dry_run(tmp_path, "public")
    assert result.returncode == 0, result.stderr
    assert "bash scripts/setup.sh" in result.stdout
    leftovers = [p.name for p in (tmp_path / "home").iterdir() if p.name.startswith(".tllm-")]
    assert leftovers == []


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell")
def test_no_argument_prints_the_three_ways_to_start(tmp_path):
    result = _dry_run(tmp_path)
    assert result.returncode == 0, result.stderr
    for word in ("create", "join", "public"):
        assert word in result.stdout
