"""Noticing that the owner wants their GPU back.

A device lends its GPU while nobody else is using it. When the owner starts a game
or a training job, the pool must back off without being asked, or lending is a
nuisance and nobody will do it.

The signal is deliberately plain: **how much GPU memory is held by processes that
are not part of the pool**. A desktop and a browser hold a few hundred MB between
them; a game or an ML job holds gigabytes. A device is "in use" when that figure
crosses a threshold, and "free" again after it has stayed below it for a while.
The two delays differ on purpose. Backing off should be quick; coming back should
wait until the owner has really finished, or the pool would flap in and out every
time a game shows its menu.

Only NVIDIA GPUs on Linux and Windows are supported for now, through
``nvidia-smi``. Where there is no such tool the monitor does nothing, and the
owner's button still works.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GpuProcess:
    pid: int
    kind: str
    sm_percent: float
    memory_mb: int
    command: str


def parse_pmon(text: str) -> list[GpuProcess]:
    """Read the table printed by ``nvidia-smi pmon -c 1 -s um``.

    Columns are positional and any of them may be ``-`` when the driver has
    nothing to report, which is common for an idle process.
    """
    found = []
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 11:
            continue

        def number(token: str) -> float:
            return float(token) if token.replace(".", "", 1).isdigit() else 0.0

        found.append(
            GpuProcess(
                pid=int(parts[1]),
                kind=parts[2],
                sm_percent=number(parts[3]),
                memory_mb=int(number(parts[9])),
                command=" ".join(parts[10:]),
            )
        )
    return found


def read_gpu_processes() -> list[GpuProcess] | None:
    """What is using the GPU right now, or ``None`` if that cannot be known."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(  # noqa: S603
            ["nvidia-smi", "pmon", "-c", "1", "-s", "um"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    return parse_pmon(out)


def belongs_to_pool(pid: int) -> bool:
    """Whether ``pid`` is one of the pool's own processes.

    Those must not count as the owner using the GPU, or a device would see its own
    shards and back off from itself. Read from the process's command line, which
    on Linux is under ``/proc``. Where that is unavailable the process is assumed
    to be someone else's, which errs toward lending less.
    """
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
    except OSError:
        return False
    return b"torrent_llm" in cmdline or b"torrent-peer" in cmdline or b"torrent-lab" in cmdline


def owner_load(
    processes: list[GpuProcess], is_pool: Callable[[int], bool] = belongs_to_pool
) -> tuple[int, float]:
    """GPU memory (MB) and busiest compute share (%) held by everyone but the pool."""
    others = [p for p in processes if not is_pool(p.pid)]
    return sum(p.memory_mb for p in others), max((p.sm_percent for p in others), default=0.0)


class OwnerMonitor:
    """Watches the GPU and tells a device when its owner needs it.

    Args:
        set_busy: Called with ``True`` when the owner is using the device and
            ``False`` when they have stopped. It should return the device's state.
        busy_mb: Other processes holding at least this much GPU memory mean "in use".
        busy_sm: ...or any one of them using at least this share of the GPU.
        busy_after: Consecutive samples over the line before backing off.
        idle_after: Consecutive samples under it before coming back.
    """

    def __init__(
        self,
        set_busy: Callable[[bool], object],
        *,
        probe: Callable[[], list[GpuProcess] | None] = read_gpu_processes,
        is_pool: Callable[[int], bool] = belongs_to_pool,
        busy_mb: int = 1500,
        busy_sm: float = 50.0,
        busy_after: int = 2,
        idle_after: int = 6,
        interval_s: float = 1.0,
    ) -> None:
        self._set_busy = set_busy
        self._probe = probe
        self._is_pool = is_pool
        self.busy_mb = busy_mb
        self.busy_sm = busy_sm
        self.busy_after = busy_after
        self.idle_after = idle_after
        self.interval_s = interval_s
        self._over = 0
        self._under = 0
        #: True only while the monitor is the reason the device is backed off, so
        #: it never undoes a decision the owner made by hand.
        self.triggered = False
        self.last_mb = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="owner-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self.sample()
            except Exception:  # noqa: BLE001 - a failed probe must not end the monitor
                logger.exception("owner probe failed")

    def sample(self) -> None:
        """Take one reading and act on it."""
        processes = self._probe()
        if processes is None:
            return
        mb, sm = owner_load(processes, self._is_pool)
        self.last_mb = mb
        in_use = mb >= self.busy_mb or sm >= self.busy_sm
        if in_use:
            self._over += 1
            self._under = 0
            if self._over >= self.busy_after and not self.triggered:
                logger.info("owner is using the GPU (%d MB held by others); backing off", mb)
                self.triggered = True
                self._set_busy(True)
        else:
            self._under += 1
            self._over = 0
            if self._under >= self.idle_after and self.triggered:
                logger.info("owner has stopped; lending the GPU again")
                self.triggered = False
                self._set_busy(False)
