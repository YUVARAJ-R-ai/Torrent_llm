"""What a device needs to know about its own machine to join a pool.

On one laptop, "127.0.0.1" and "lend whatever" are fine. On several real machines
each device has to tell the others an address they can actually reach and a memory
budget it can actually afford, and nobody should have to work either out by hand.
Everything here uses only the standard library, so it works the same on Windows,
Linux and macOS without installing anything extra.
"""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path


def lan_address() -> str:
    """The address other devices on this network would use to reach this one.

    Found by asking the operating system which interface it would use to reach the
    wider network. Connecting a UDP socket sends nothing; it only makes the system
    choose a route, and the socket then reports the local address it picked. That is
    right on a laptop with Wi-Fi, Ethernet and a VPN all at once, where guessing from
    the hostname often gives the wrong one. Falls back to loopback with no network.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # A documentation address (RFC 5737): never routed, never answers, and
        # nothing is transmitted. It only has to look like "somewhere else".
        probe.connect(("192.0.2.1", 9))
        return probe.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        probe.close()


def tailscale_address() -> str | None:
    """This device's Tailscale address, if Tailscale is installed and running."""
    tool = shutil.which("tailscale")
    if tool is None:
        return None
    try:
        out = subprocess.run(  # noqa: S603
            [tool, "ip", "-4"], capture_output=True, text=True, timeout=5, check=True
        ).stdout.split()
    except (subprocess.SubprocessError, OSError):
        return None
    return out[0] if out else None


def choose_host(preference: str = "auto") -> str:
    """Resolve ``--host``: an explicit address, ``auto``, ``lan`` or ``tailscale``."""
    if preference in ("auto", "lan"):
        return lan_address()
    if preference == "tailscale":
        found = tailscale_address()
        if found is None:
            raise RuntimeError(
                "asked for the Tailscale address but Tailscale is not running here. "
                "Install it from https://tailscale.com/download and sign in, or use --host lan."
            )
        return found
    return preference


def available_ram_mb() -> int:
    """Memory the system could give a new program right now, in MB.

    ``/proc/meminfo`` on Linux, the Win32 call on Windows, and total memory on macOS
    (where the kernel does not report a comparable "available" figure without extra
    tools, so the share lent later is kept conservative).
    """
    system = platform.system()
    if system == "Linux":
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) // 1024
    if system == "Windows":

        class Status(ctypes.Structure):
            _fields_ = [  # noqa: RUF012
                ("length", ctypes.c_ulong),
                ("load", ctypes.c_ulong),
                ("total_phys", ctypes.c_ulonglong),
                ("avail_phys", ctypes.c_ulonglong),
                ("total_page", ctypes.c_ulonglong),
                ("avail_page", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("avail_virtual", ctypes.c_ulonglong),
                ("avail_extended", ctypes.c_ulonglong),
            ]

        status = Status()
        status.length = ctypes.sizeof(Status)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))  # type: ignore[attr-defined]
        return int(status.avail_phys // 1024**2)
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        return int(total // 1024**2 // 2)
    except (ValueError, OSError, AttributeError):
        return 4096


def free_vram_mb(device_index: int = 0) -> int | None:
    """Free memory on the first NVIDIA GPU, or ``None`` if there is not one."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info(device_index)
        return int(free // 1024**2)
    except Exception:  # noqa: BLE001 - a broken driver must not stop the device starting
        return None


#: Share of what is free that a device lends by default. Generous enough to be
#: useful, small enough that the owner can still use their machine.
LEND_SHARE = {"gpu": 0.7, "cpu": 0.4}


def auto_memory_mb(device: str) -> int:
    """How much to lend the pool when the owner has not said.

    On a GPU that is most of its free memory, since the model has to fit there. On a
    CPU it is a smaller share of RAM, because the owner is probably running other
    things in the same memory. Never negative.
    """
    if device.startswith("cuda"):
        vram = free_vram_mb()
        if vram is not None:
            return max(0, int(vram * LEND_SHARE["gpu"]))
    return max(0, int(available_ram_mb() * LEND_SHARE["cpu"]))


def port_is_free(port: int, host: str = "0.0.0.0") -> bool:
    """Whether a TCP port can be listened on right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # No SO_REUSEADDR here: it would make a port that is in use look free on
        # some systems, which is the one answer this check must not get wrong.
        try:
            sock.bind((host, port))
        except OSError:
            return False
        return True


def firewall_advice(control_port: int, shard_ports: tuple[int, int] | None) -> str:
    """The command that lets other devices in, for this operating system."""
    span = (
        f"{control_port}"
        if shard_ports is None
        else f"{control_port},{shard_ports[0]}-{shard_ports[1]}"
    )
    system = platform.system()
    if system == "Windows":
        return (
            "In an Administrator PowerShell:\n"
            f'  New-NetFirewallRule -DisplayName "Torrent-LLM" -Direction Inbound -Protocol TCP '
            f"-LocalPort {span} -Action Allow -Profile Private\n"
            '  New-NetFirewallRule -DisplayName "Torrent-LLM discovery" -Direction Inbound '
            "-Protocol UDP -LocalPort 9099 -Action Allow -Profile Private\n"
            "or choose 'Allow' when Windows asks about python.exe on a private network."
        )
    if system == "Darwin":
        return (
            "macOS asks 'Do you want the application python to accept incoming network "
            "connections?' the first time. Choose Allow."
        )
    return (
        f"If a firewall is on, allow TCP {span} and UDP 9099, for example:\n"
        f"  sudo ufw allow {span.replace('-', ':')}/tcp && sudo ufw allow 9099/udp"
        if shutil.which("ufw")
        else f"If a firewall is on, allow TCP {span} and UDP 9099 from your local network."
    )


def describe_machine() -> str:
    """One line saying what this machine is, for the start-up banner."""
    python = sys.version.split()[0]
    return f"{platform.node()} ({platform.system()} {platform.machine()}, Python {python})"
