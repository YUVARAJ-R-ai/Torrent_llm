"""Tests for cross-platform process groups (torrent_llm.proc).

Verifies starting, terminating, and force-killing process groups across
POSIX and Windows platforms.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

from torrent_llm.proc import kill_group, popen_group, terminate_group


def test_popen_and_terminate_group() -> None:
    """Start a sleeping process with popen_group and terminate it cleanly."""
    cmd = [sys.executable, "-c", "import time; time.sleep(60)"]
    proc = popen_group(cmd)

    ret = terminate_group(proc, timeout=5.0)

    assert proc.poll() is not None
    assert isinstance(ret, int)


def test_popen_group_rejects_shell_true() -> None:
    """popen_group must raise ValueError if shell=True is passed."""
    with pytest.raises(ValueError, match="shell=True"):
        popen_group(["echo", "1"], shell=True)


@pytest.mark.skipif(sys.platform == "win32", reason="SIGTERM ignore is POSIX only")
def test_child_ignoring_sigterm_is_force_killed() -> None:
    """A process ignoring SIGTERM must be forcefully killed after timeout."""
    cmd = [
        sys.executable,
        "-c",
        (
            "import signal, sys, time; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "sys.stdout.write('ready\\n'); sys.stdout.flush(); "
            "time.sleep(60)"
        ),
    ]
    proc = popen_group(cmd, stdout=subprocess.PIPE, text=True)
    assert proc.stdout is not None
    assert proc.stdout.readline() == "ready\n"

    ret = terminate_group(proc, timeout=1.0)

    assert proc.poll() is not None
    assert ret == -signal.SIGKILL


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process group inspection")
def test_grandchild_is_also_gone_after_terminate_group() -> None:
    """Terminating a process group must terminate all descendants in the group."""
    cmd = [
        sys.executable,
        "-c",
        (
            "import subprocess, sys, time; "
            "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
            "sys.stdout.write(f'{p.pid}\\n'); sys.stdout.flush(); "
            "time.sleep(60)"
        ),
    ]
    proc = popen_group(cmd, stdout=subprocess.PIPE, text=True)
    assert proc.stdout is not None
    grandchild_pid = int(proc.stdout.readline().strip())
    assert os.kill(grandchild_pid, 0) is None

    terminate_group(proc, timeout=5.0)

    assert proc.poll() is not None
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            os.kill(grandchild_pid, 0)
            time.sleep(0.05)
        except ProcessLookupError:
            break

    with pytest.raises(ProcessLookupError):
        os.kill(grandchild_pid, 0)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process group inspection")
def test_grandchild_ignoring_sigterm_is_force_killed() -> None:
    """When a child exits on SIGTERM but a grandchild ignores SIGTERM, the grandchild is killed."""
    cmd = [
        sys.executable,
        "-c",
        (
            "import subprocess, sys, time; "
            "p = subprocess.Popen(["
            "    sys.executable, '-c', "
            "    'import signal, time; ' "
            "    'signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'"
            "]); "
            "sys.stdout.write(f'{p.pid}\\n'); sys.stdout.flush(); "
            "time.sleep(60)"
        ),
    ]
    proc = popen_group(cmd, stdout=subprocess.PIPE, text=True)
    assert proc.stdout is not None
    grandchild_pid = int(proc.stdout.readline().strip())
    assert os.kill(grandchild_pid, 0) is None

    terminate_group(proc, timeout=1.0)
    assert proc.poll() is not None

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            os.kill(grandchild_pid, 0)
            time.sleep(0.05)
        except ProcessLookupError:
            break

    with pytest.raises(ProcessLookupError):
        os.kill(grandchild_pid, 0)


def test_terminate_group_on_already_finished_process() -> None:
    """Calling terminate_group on an already exited process returns its exit code."""
    cmd = [sys.executable, "-c", "import sys; sys.exit(42)"]
    proc = popen_group(cmd)
    proc.wait(timeout=5.0)
    assert proc.poll() == 42

    ret = terminate_group(proc, timeout=5.0)

    assert ret == 42


def test_kill_group_force_stops_process() -> None:
    """kill_group immediately forces a process to terminate."""
    cmd = [sys.executable, "-c", "import time; time.sleep(60)"]
    proc = popen_group(cmd)

    kill_group(proc)

    assert proc.poll() is not None


def test_kill_group_does_not_raise_when_the_process_will_not_exit(caplog) -> None:
    """A process stuck after SIGKILL is logged, not raised, so teardown carries on."""
    mock_proc = MagicMock()
    mock_proc.pid = 12345
    mock_proc.poll.return_value = None
    mock_proc.wait.side_effect = subprocess.TimeoutExpired(cmd="fake", timeout=0.1)

    # Never send a real signal from a test with a fake pid.
    with (
        patch("torrent_llm.proc.os.killpg") as killpg,
        patch("torrent_llm.proc._IS_WINDOWS", False),
    ):
        kill_group(mock_proc, timeout=0.1)
    killpg.assert_called_once()
    assert "did not exit" in caplog.text


@pytest.mark.parametrize("pid", [0, 1, -1, None, True])
def test_a_pid_of_0_or_1_or_a_non_integer_is_never_signalled(pid) -> None:
    """killpg(1) is kill(-1): it would hit every process of this user."""
    mock_proc = MagicMock()
    mock_proc.pid = pid
    mock_proc.poll.return_value = None
    with (
        patch("torrent_llm.proc.os.killpg") as killpg,
        patch("torrent_llm.proc._IS_WINDOWS", False),
    ):
        with pytest.raises(ValueError, match="refusing"):
            kill_group(mock_proc)
        with pytest.raises(ValueError, match="refusing"):
            terminate_group(mock_proc)
    killpg.assert_not_called()


def test_an_unset_mock_pid_is_refused_too() -> None:
    """A MagicMock's pid converts to 1; it must be refused, not signalled."""
    with (
        patch("torrent_llm.proc.os.killpg") as killpg,
        patch("torrent_llm.proc._IS_WINDOWS", False),
    ):
        with pytest.raises(ValueError):
            kill_group(MagicMock())
    killpg.assert_not_called()


def test_windows_paths_mocked() -> None:
    """Verify Windows-specific branches with mocks when not running on Windows."""
    with patch("torrent_llm.proc._IS_WINDOWS", True):
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_popen.return_value = mock_proc
            popen_group(["cmd.exe", "/c", "echo 1"], creationflags=0x10)
            assert mock_popen.called
            call_kwargs = mock_popen.call_args[1]
            assert call_kwargs["creationflags"] & 0x00000200
            assert call_kwargs["shell"] is False

        mock_proc = MagicMock()
        mock_proc.poll.side_effect = [None, None]
        mock_proc.wait.side_effect = [subprocess.TimeoutExpired(cmd="test", timeout=1.0), 0, 0]
        mock_proc.pid = 12345

        with patch("subprocess.run") as mock_run:
            ret = terminate_group(mock_proc, timeout=1.0)
            mock_proc.send_signal.assert_called_once()
            mock_run.assert_called_once()
            cmd_args = mock_run.call_args[0][0]
            assert cmd_args == ["taskkill", "/T", "/F", "/PID", "12345"]
            assert ret == 0

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 12345
        with patch("subprocess.run") as mock_run:
            kill_group(mock_proc)
            mock_run.assert_called_once()
            cmd_args = mock_run.call_args[0][0]
            assert cmd_args == ["taskkill", "/T", "/F", "/PID", "12345"]
