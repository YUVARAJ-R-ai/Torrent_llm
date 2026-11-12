"""Where a device keeps its files, on every operating system, without extra dependencies.

=========  ==================================================================
Windows    ``%LOCALAPPDATA%\\TorrentLLM`` (fallback ``~/AppData/Local/TorrentLLM``)
macOS      ``~/Library/Application Support/TorrentLLM``
Linux/BSD  ``$XDG_DATA_HOME/torrent-llm`` (fallback ``~/.local/share/torrent-llm``)
=========  ==================================================================

``TORRENT_LLM_HOME`` overrides all of them (tests and portable installs use it).
Files inside: ``identity.json`` (the device key), ``admin-token`` (the local API
secret), ``settings.json``, ``swarms/`` (joined invite swarms), ``trust.json``
(reputation and bans), ``peers.json`` (known peer records), ``models/`` (shared
model chunks).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ENV_OVERRIDE = "TORRENT_LLM_HOME"


def data_dir(
    *, platform: str | None = None, env: dict[str, str] | None = None, home: Path | None = None
) -> Path:
    """The data directory for ``platform`` (default ``sys.platform``). Does not create it.

    ``env`` and ``home`` default to ``os.environ`` and ``Path.home()``; tests pass
    their own so every platform's rule is checkable on any machine.
    """
    effective_env = os.environ if env is None else env
    override = effective_env.get(ENV_OVERRIDE)
    if override:
        return Path(override)

    effective_platform = sys.platform if platform is None else platform
    effective_home = Path.home() if home is None else Path(home)

    if effective_platform == "win32":
        local_app_data = effective_env.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / "TorrentLLM"
        return effective_home / "AppData" / "Local" / "TorrentLLM"

    if effective_platform == "darwin":
        return effective_home / "Library" / "Application Support" / "TorrentLLM"

    xdg_data_home = effective_env.get("XDG_DATA_HOME")
    if xdg_data_home:
        xdg_path = Path(xdg_data_home)
        if xdg_path.is_absolute():
            return xdg_path / "torrent-llm"

    return effective_home / ".local" / "share" / "torrent-llm"


def ensure_data_dir(**kwargs) -> Path:
    """:func:`data_dir`, created (with parents) if missing."""
    d = data_dir(**kwargs)
    d.mkdir(parents=True, exist_ok=True)
    return d


def identity_path(**kwargs) -> Path:
    """The path to the device's key pair (``identity.json``)."""
    return data_dir(**kwargs) / "identity.json"


def admin_token_path(**kwargs) -> Path:
    """The path to the local API secret token (``admin-token``)."""
    return data_dir(**kwargs) / "admin-token"


def settings_path(**kwargs) -> Path:
    """The path to user configuration (``settings.json``)."""
    return data_dir(**kwargs) / "settings.json"


def trust_path(**kwargs) -> Path:
    """The path to reputation and bans (``trust.json``)."""
    return data_dir(**kwargs) / "trust.json"


def peers_path(**kwargs) -> Path:
    """The path to cached peer records (``peers.json``)."""
    return data_dir(**kwargs) / "peers.json"


def swarms_dir(**kwargs) -> Path:
    """The directory holding joined invite swarms (``swarms/``)."""
    return data_dir(**kwargs) / "swarms"


def models_dir(**kwargs) -> Path:
    """The directory holding shared model chunks (``models/``)."""
    return data_dir(**kwargs) / "models"


__all__ = [
    "ENV_OVERRIDE",
    "admin_token_path",
    "data_dir",
    "ensure_data_dir",
    "identity_path",
    "models_dir",
    "peers_path",
    "settings_path",
    "swarms_dir",
    "trust_path",
]
