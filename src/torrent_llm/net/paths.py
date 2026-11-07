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

APP_DIR_NAME = {"win32": "TorrentLLM", "darwin": "TorrentLLM"}
ENV_OVERRIDE = "TORRENT_LLM_HOME"


def data_dir(
    *, platform: str | None = None, env: dict[str, str] | None = None, home: Path | None = None
) -> Path:
    """The data directory for ``platform`` (default ``sys.platform``). Does not create it.

    ``env`` and ``home`` default to ``os.environ`` and ``Path.home()``; tests pass
    their own so every platform's rule is checkable on any machine.
    """
    raise NotImplementedError


def ensure_data_dir(**kwargs) -> Path:
    """:func:`data_dir`, created (with parents) if missing."""
    raise NotImplementedError


def identity_path(**kwargs) -> Path:
    raise NotImplementedError


def admin_token_path(**kwargs) -> Path:
    raise NotImplementedError


def settings_path(**kwargs) -> Path:
    raise NotImplementedError


def trust_path(**kwargs) -> Path:
    raise NotImplementedError


def peers_path(**kwargs) -> Path:
    raise NotImplementedError


def swarms_dir(**kwargs) -> Path:
    raise NotImplementedError


def models_dir(**kwargs) -> Path:
    raise NotImplementedError


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
