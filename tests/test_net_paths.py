"""Tests for per-operating-system data and config directory paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from torrent_llm.net.paths import (
    ENV_OVERRIDE,
    admin_token_path,
    data_dir,
    ensure_data_dir,
    identity_path,
    models_dir,
    peers_path,
    settings_path,
    swarms_dir,
    trust_path,
)


def test_windows_data_dir_with_localappdata(tmp_path: Path) -> None:
    custom_appdata = tmp_path / "LocalApp"
    res = data_dir(
        platform="win32",
        env={"LOCALAPPDATA": str(custom_appdata)},
        home=tmp_path,
    )
    assert res == custom_appdata / "TorrentLLM"


def test_windows_data_dir_fallback(tmp_path: Path) -> None:
    res = data_dir(platform="win32", env={}, home=tmp_path)
    assert res == tmp_path / "AppData" / "Local" / "TorrentLLM"


def test_macos_data_dir(tmp_path: Path) -> None:
    res = data_dir(platform="darwin", env={}, home=tmp_path)
    assert res == tmp_path / "Library" / "Application Support" / "TorrentLLM"


def test_linux_data_dir_with_xdg(tmp_path: Path) -> None:
    custom_xdg = tmp_path / "custom_share"
    res = data_dir(
        platform="linux",
        env={"XDG_DATA_HOME": str(custom_xdg)},
        home=tmp_path,
    )
    assert res == custom_xdg / "torrent-llm"


def test_linux_data_dir_fallback(tmp_path: Path) -> None:
    res = data_dir(platform="linux", env={}, home=tmp_path)
    assert res == tmp_path / ".local" / "share" / "torrent-llm"


def test_bsd_data_dir_fallback(tmp_path: Path) -> None:
    res = data_dir(platform="freebsd", env={}, home=tmp_path)
    assert res == tmp_path / ".local" / "share" / "torrent-llm"


def test_override_wins_on_every_platform(tmp_path: Path) -> None:
    override_path = tmp_path / "custom_home"
    env = {
        ENV_OVERRIDE: str(override_path),
        "LOCALAPPDATA": str(tmp_path / "ignored_win"),
        "XDG_DATA_HOME": str(tmp_path / "ignored_xdg"),
    }
    for plat in ("win32", "darwin", "linux", "freebsd", "openbsd", "sunos5"):
        assert data_dir(platform=plat, env=env, home=tmp_path) == override_path


def test_empty_string_env_values_are_ignored(tmp_path: Path) -> None:
    # Empty TORRENT_LLM_HOME falls through to platform default
    assert (
        data_dir(
            platform="win32",
            env={ENV_OVERRIDE: ""},
            home=tmp_path,
        )
        == tmp_path / "AppData" / "Local" / "TorrentLLM"
    )

    assert (
        data_dir(
            platform="darwin",
            env={ENV_OVERRIDE: ""},
            home=tmp_path,
        )
        == tmp_path / "Library" / "Application Support" / "TorrentLLM"
    )

    assert (
        data_dir(
            platform="linux",
            env={ENV_OVERRIDE: ""},
            home=tmp_path,
        )
        == tmp_path / ".local" / "share" / "torrent-llm"
    )

    # Empty LOCALAPPDATA falls back to home AppData
    assert (
        data_dir(
            platform="win32",
            env={"LOCALAPPDATA": ""},
            home=tmp_path,
        )
        == tmp_path / "AppData" / "Local" / "TorrentLLM"
    )

    # Empty XDG_DATA_HOME falls back to home .local/share
    assert (
        data_dir(
            platform="linux",
            env={"XDG_DATA_HOME": ""},
            home=tmp_path,
        )
        == tmp_path / ".local" / "share" / "torrent-llm"
    )


def test_relative_xdg_data_home_ignored(tmp_path: Path) -> None:
    for rel_path in ("relative/path", ".local/share", "./custom", "share"):
        assert (
            data_dir(
                platform="linux",
                env={"XDG_DATA_HOME": rel_path},
                home=tmp_path,
            )
            == tmp_path / ".local" / "share" / "torrent-llm"
        )


def test_ensure_data_dir_creates_directory(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "deep" / "data"
    env = {ENV_OVERRIDE: str(target)}
    assert not target.exists()

    created = ensure_data_dir(env=env)
    assert created == target
    assert created.is_dir()

    # Calling again is idempotent and does not raise
    again = ensure_data_dir(env=env)
    assert again == target
    assert again.is_dir()


def test_helper_paths_are_inside_data_dir(tmp_path: Path) -> None:
    base = tmp_path / "base"
    kw = {"env": {ENV_OVERRIDE: str(base)}}

    expected_helpers = {
        identity_path: "identity.json",
        admin_token_path: "admin-token",
        settings_path: "settings.json",
        trust_path: "trust.json",
        peers_path: "peers.json",
        swarms_dir: "swarms",
        models_dir: "models",
    }

    for fn, filename in expected_helpers.items():
        p = fn(**kw)
        assert p == base / filename
        assert p.parent == base
        # Helper paths do NOT create anything
        assert not p.exists()


def test_default_arguments(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # When ENV_OVERRIDE is present in the actual environment
    custom_dir = tmp_path / "my_torrent_llm"
    monkeypatch.setenv(ENV_OVERRIDE, str(custom_dir))
    assert data_dir() == custom_dir
    assert identity_path() == custom_dir / "identity.json"

    # When ENV_OVERRIDE is removed, verify default call returns a Path
    monkeypatch.delenv(ENV_OVERRIDE, raising=False)
    default_dir = data_dir()
    assert isinstance(default_dir, Path)
    assert default_dir.is_absolute()
