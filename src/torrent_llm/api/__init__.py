"""HTTP instrumentation over the shard chain.

Importing this package requires the ``api`` extra::

    uv pip install -e ".[api]"

It is an optional extra rather than a core dependency because a shard node
should not need a web framework installed just to host layers.
"""

try:
    from torrent_llm.api.app import Tokenizer, create_app
except ImportError as exc:  # pragma: no cover - only without the extra installed
    raise ImportError(
        "the HTTP layer needs the 'api' extra. Install it with:\n\n"
        '    uv pip install -e ".[api]"\n'
    ) from exc

__all__ = ["Tokenizer", "create_app"]
