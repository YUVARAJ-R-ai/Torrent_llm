"""Generate the gRPC stubs from ``activation.proto``.

The stubs are generated rather than committed, so the proto stays the single
source of truth and nobody edits a checked-in ``_pb2.py`` by accident. Run::

    python -m torrent_llm.transport.codegen

Generation is rooted at ``src/`` so protoc emits a package-qualified import
(``from torrent_llm.transport import activation_pb2``) rather than the flat
``import activation_pb2`` that breaks inside a package.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROTO_RELATIVE = Path("torrent_llm/transport/activation.proto")


def source_root() -> Path:
    """The ``src/`` directory, whether running from a checkout or an editable install."""
    return Path(__file__).resolve().parents[2]


def generate() -> None:
    """Run protoc. Raises CalledProcessError if generation fails."""
    root = source_root()
    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"--proto_path={root}",
        f"--python_out={root}",
        f"--pyi_out={root}",
        f"--grpc_python_out={root}",
        str(PROTO_RELATIVE),
    ]
    subprocess.run(cmd, check=True)
    print(f"generated stubs in {root / PROTO_RELATIVE.parent}")


if __name__ == "__main__":
    generate()
