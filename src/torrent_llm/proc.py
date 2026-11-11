"""Cross-platform process groups for subprocess management.

When running processes that may spawn subprocesses of their own (such as peer
nodes or workers), stopping only the top-level process orphans its children.
To clean up reliably across platforms:

* On POSIX systems, a new session is started so the process becomes a process
  group leader. Signals (SIGTERM, SIGKILL) are sent to the entire group using
  os.killpg.
* On Windows systems, processes are started with CREATE_NEW_PROCESS_GROUP,
  notified via CTRL_BREAK_EVENT, and forcefully stopped with taskkill /T /F.

This module unifies these patterns behind popen_group, terminate_group, and
kill_group.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
from typing import Any

logger = logging.getLogger(__name__)

_IS_WINDOWS = sys.platform == "win32"


def _group_id(proc: subprocess.Popen) -> int:
    """The process group to signal, refusing anything that is not a real child's pid.

    ``os.killpg(1, sig)`` is ``kill(-1, sig)``: every process this user owns. A mock
    process (whose ``pid`` turns into 1) once took a whole desktop session down that
    way, so only a plain integer above 1 is ever signalled.
    """
    pid = proc.pid
    if type(pid) is not int or pid <= 1:
        raise ValueError(f"refusing to signal process group {pid!r}")
    return pid


def popen_group(cmd: list[str], **kwargs: Any) -> subprocess.Popen:
    """Start a command in a new process group or session.

    On POSIX systems, creates a new session so signals can target the entire group.
    On Windows systems, sets the CREATE_NEW_PROCESS_GROUP creation flag.
    Passes all other keyword arguments through to :class:`subprocess.Popen`.
    Never uses shell=True.
    """
    if kwargs.get("shell"):
        raise ValueError("popen_group does not allow shell=True")
    kwargs["shell"] = False

    if _IS_WINDOWS:
        flags = kwargs.get("creationflags", 0)
        flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        kwargs["creationflags"] = flags
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(cmd, **kwargs)  # noqa: S603


def terminate_group(proc: subprocess.Popen, *, timeout: float = 10.0) -> int:
    """Ask the whole group to stop, wait up to timeout, then force it.

    Returns the exit code. On POSIX, sends SIGTERM then SIGKILL. On Windows,
    sends CTRL_BREAK_EVENT then runs taskkill to force termination.
    Already-exited processes and ProcessLookupError are handled cleanly.
    """
    if proc.poll() is not None:
        return proc.returncode

    if _IS_WINDOWS:
        try:
            proc.send_signal(getattr(signal, "CTRL_BREAK_EVENT", signal.SIGTERM))
        except (ProcessLookupError, OSError):
            pass
    else:
        try:
            os.killpg(_group_id(proc), signal.SIGTERM)
        except ProcessLookupError:
            pass

    try:
        ret = proc.wait(timeout=timeout)
        if not _IS_WINDOWS:
            try:
                os.killpg(_group_id(proc), signal.SIGKILL)
            except ProcessLookupError:
                pass
        return ret
    except subprocess.TimeoutExpired:
        pass

    kill_group(proc, timeout=timeout)
    return proc.wait()


def kill_group(proc: subprocess.Popen, *, timeout: float = 5.0) -> None:
    """Immediately force-kill the process group without waiting for a clean exit."""
    if _IS_WINDOWS:
        try:
            subprocess.run(  # noqa: S603
                ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                capture_output=True,
                check=False,
            )
        except (ProcessLookupError, OSError):
            pass
    else:
        try:
            os.killpg(_group_id(proc), signal.SIGKILL)
        except ProcessLookupError:
            pass

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        # SIGKILL cannot be refused; a process still here is stuck in the kernel (e.g. a
        # hung disk). Say so and move on rather than fail the caller's teardown.
        logger.warning("process %s did not exit within %.0fs of SIGKILL", proc.pid, timeout)
