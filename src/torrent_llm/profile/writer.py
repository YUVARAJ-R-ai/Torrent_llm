"""Recording hops to disk.

JSONL, one record per line, metadata first. Appending line-by-line rather than
buffering means a run that dies halfway still leaves usable measurements, which
matters when a rig session is a scheduled slot on someone else's machine.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from types import TracebackType

from torrent_llm.profile.records import HopRecord, RunMetadata
from torrent_llm.transport import HopResult


class HopProfiler:
    """Collects :class:`HopRecord` rows, optionally streaming them to a file."""

    def __init__(
        self,
        *,
        path: str | Path | None = None,
        metadata: RunMetadata | None = None,
    ) -> None:
        self.records: list[HopRecord] = []
        self.path = Path(path) if path else None
        self._handle = None

        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("w", encoding="utf-8")
            if metadata is not None:
                self._handle.write(metadata.to_json() + "\n")
                self._handle.flush()

    def record(
        self,
        result: HopResult,
        *,
        request_id: str,
        address: str,
        shape: tuple[int, ...],
        dtype: str,
        phase: str = "prefill",
        **extra: object,
    ) -> HopRecord:
        """Turn a transport-level result into a durable record.

        ``shape`` is the *logical* shape of what was sent. For hop 0 that is
        token ids ``(batch, seq)``, so hidden_size is 0 — the embedding has not
        happened yet and there is no hidden dimension to report.
        """
        batch = shape[0] if shape else 1
        seq_len = shape[1] if len(shape) > 1 else 0
        hidden_size = shape[2] if len(shape) > 2 else 0

        record = HopRecord(
            request_id=request_id,
            hop=result.hop,
            codec=result.codec,
            address=address,
            batch=batch,
            seq_len=seq_len,
            hidden_size=hidden_size,
            dtype=dtype,
            sent_bytes=result.sent_bytes,
            received_bytes=result.received_bytes,
            uncompressed_bytes=result.sent_uncompressed_bytes,
            wall_ns=result.wall_ns,
            compute_ns=result.compute_ns,
            phase=phase,
            extra=dict(extra),
        )
        self.records.append(record)
        if self._handle is not None:
            self._handle.write(record.to_json() + "\n")
            self._handle.flush()
        return record

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> HopProfiler:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def load_records(path: str | Path) -> tuple[RunMetadata | None, list[HopRecord]]:
    """Read a profile file back.

    Unknown keys are dropped rather than raising, so records written by an older
    version of the schema still load. Silently *adding* defaults for missing
    required fields is not done — a truncated record should fail loudly.
    """
    import json

    metadata: RunMetadata | None = None
    records: list[HopRecord] = []
    meta_fields = set(RunMetadata.__dataclass_fields__)
    hop_fields = set(HopRecord.__dataclass_fields__)

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.pop("record_type", None) == "metadata":
            metadata = RunMetadata(**{k: v for k, v in payload.items() if k in meta_fields})
        else:
            records.append(HopRecord(**{k: v for k, v in payload.items() if k in hop_fields}))
    return metadata, records
