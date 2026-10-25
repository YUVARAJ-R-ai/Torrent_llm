#!/usr/bin/env python3
"""Run one model across several machines, whatever OS each one runs.

Every machine gets the same topology file. Each then plays one of two roles:

    python scripts/cluster.py worker --config configs/cluster-2laptop.yaml --index 1
    python scripts/cluster.py head   --config configs/cluster-2laptop.yaml --local 0

A *worker* hosts one shard and waits. The *head* hosts its own shard(s), waits
until every remote shard answers, then starts the HTTP API and the dashboard and
warms the model up. ``check`` asks every node what it hosts without starting
anything, which is the first thing to run when something will not connect:

    python scripts/cluster.py check --config configs/cluster-2laptop.yaml

This is ``scripts/demo.sh`` rewritten in Python so it behaves the same on
Windows, Linux and macOS: no bash, no ``pgrep``, no ``curl``. It finds the
project venv by itself (``$TORRENT_VENV``, default ``~/Coding/venvs/torrent-llm``)
so a plain ``python`` is enough to start it. Logs go to
``<system temp>/torrent-demo``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
IS_WINDOWS = os.name == "nt"
LOGS = Path(os.environ.get("TORRENT_DEMO_LOGS", Path(tempfile.gettempdir()) / "torrent-demo"))


# ---------------------------------------------------------------------------
# environment


def _reexec_in_venv() -> None:
    """Re-run this script under the project venv if the current Python lacks it.

    Saves remembering to activate the venv first, and the activation command
    differs on every shell (bash, fish, PowerShell, cmd).
    """
    try:
        import torrent_llm  # noqa: F401

        return
    except ImportError:
        pass
    venv = Path(os.environ.get("TORRENT_VENV", Path.home() / "Coding" / "venvs" / "torrent-llm"))
    python = venv / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")
    if not python.exists() or Path(sys.prefix).resolve() == venv.resolve():
        sys.exit(
            "torrent_llm is not installed for this Python and no venv was found at "
            f"{venv}.\nRun scripts/setup.sh (Linux/macOS) or scripts/setup.ps1 (Windows) "
            "first, or set TORRENT_VENV to where the venv lives."
        )
    # Not subprocess.call: on Ctrl+C it kills the child after a quarter second,
    # cutting the head's own cleanup short and orphaning everything it started.
    # The child sees the same Ctrl+C from the terminal, so the wrapper only
    # waits; other ways of ending the wrapper are passed on to the child.
    child = subprocess.Popen([str(python), *sys.argv])
    for name in ("SIGTERM", "SIGHUP"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), lambda signum, _: child.send_signal(signum))
    while True:
        try:
            sys.exit(child.wait())
        except KeyboardInterrupt:
            continue


def _ensure_stubs() -> None:
    """Generate the gRPC stubs on a fresh checkout, as demo.sh does."""
    try:
        import torrent_llm.transport  # noqa: F401
    except ImportError:
        print("generating gRPC stubs (first run on this checkout)")
        subprocess.run([sys.executable, "-m", "torrent_llm.codegen"], check=True)


def _console_script(name: str) -> str:
    """Path to one of the package's commands, next to the venv's Python."""
    folder = Path(sys.executable).parent
    for candidate in (folder / f"{name}.exe", folder / name):
        if candidate.exists():
            return str(candidate)
    found = shutil.which(name)
    if found is None:
        sys.exit(f"{name} not found; is the package installed (pip install -e .)?")
    return found


def local_addresses() -> list[str]:
    """This machine's IPv4 addresses, best effort and without extra packages."""
    found: set[str] = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.add(str(info[4][0]))
    except OSError:
        pass
    # The address the OS would route outward from. Connecting a UDP socket
    # sends nothing; it only makes the OS choose an interface.
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.connect(("10.255.255.255", 1))
            found.add(probe.getsockname()[0])
        except OSError:
            pass
    tailscale = shutil.which("tailscale")
    if tailscale:
        try:
            out = subprocess.run(
                [tailscale, "ip", "-4"], capture_output=True, text=True, timeout=3
            ).stdout
            found.update(line.strip() for line in out.splitlines() if line.strip())
        except (OSError, subprocess.TimeoutExpired):
            pass
    return sorted(a for a in found if not a.startswith("127."))


def firewall_hint(port: int) -> str:
    """How to let peers reach ``port`` on this OS."""
    system = platform.system()
    if system == "Windows":
        return (
            "Windows Firewall: click 'Allow access' on the first prompt and tick BOTH\n"
            "Private and Public (a phone hotspot is usually 'Public'). Or, in an\n"
            "Administrator PowerShell:\n"
            f'  New-NetFirewallRule -DisplayName "Torrent-LLM {port}" -Direction Inbound '
            f"-Protocol TCP -LocalPort {port} -Action Allow -Profile Any"
        )
    if system == "Darwin":
        return "macOS: click 'Allow' when asked whether Python may accept incoming connections."
    return (
        f"Linux firewall, if one is on:  ufw: sudo ufw allow {port}/tcp\n"
        f"  firewalld: sudo firewall-cmd --add-port={port}/tcp\n"
        f"  NixOS: networking.firewall.allowedTCPPorts = [ {port} ];\n"
        '         (or trustedInterfaces = [ "tailscale0" ] when using Tailscale)'
    )


# ---------------------------------------------------------------------------
# checking the chain


def check_nodes(config, *, quiet: bool = False) -> tuple[bool, bool]:
    """Ask every node what it hosts.

    Returns ``(all_up, fatal)``. ``fatal`` means a node answered but is serving
    the wrong thing -- waiting longer will not fix that, so callers should stop.
    """
    from torrent_llm.codec import get_codec
    from torrent_llm.transport import ShardClient

    plan = config.shard_plan()
    all_up, fatal = True, False
    for i, (node, spec) in enumerate(zip(config.nodes, plan, strict=True)):
        status, detail = _probe(node.host, node.port)
        if status == "up":
            try:
                with ShardClient(node.address, get_codec("raw"), timeout=10) as client:
                    info = client.info()
            except Exception as exc:  # noqa: BLE001 - any failure here is "not ready"
                status, detail = "error", f"port open but the shard did not answer: {exc}"
            else:
                problem = _mismatch(config, spec, info)
                if problem:
                    status, detail, fatal = "WRONG", problem, True
                else:
                    detail = (
                        f"layers [{info.layer_start}:{info.layer_end})  on "
                        f"{info.hostname or '?'} ({info.os or '?'}, {info.device})"
                    )
        all_up = all_up and status == "up"
        if not quiet or status != "up":
            print(f"  shard {i}  {node.address:<22} {status:<6} {detail}")
    return all_up, fatal


def _probe(host: str, port: int) -> tuple[str, str]:
    """A plain TCP connect, which tells a closed port from an unreachable host."""
    try:
        with socket.create_connection((host, port), timeout=3):
            return "up", ""
    except ConnectionRefusedError:
        return "down", "machine reachable but nothing listening -- is the worker running?"
    except TimeoutError:
        return "down", (
            "no answer -- wrong IP, not on the same network, Wi-Fi client isolation, "
            f"or that machine's firewall is blocking TCP {port}"
        )
    except OSError as exc:
        return "down", str(exc)


def _mismatch(config, spec, info) -> str:
    """Why a node that answered cannot be used, or an empty string."""
    if info.model_id != config.model_id:
        return f"serving {info.model_id}, but the config says {config.model_id}"
    if (info.layer_start, info.layer_end) != (spec.start, spec.end):
        return (
            f"serving layers [{info.layer_start}:{info.layer_end}) but this config "
            f"plans [{spec.start}:{spec.end}) -- the machines have different config files"
        )
    return ""


# ---------------------------------------------------------------------------
# process management


class Processes:
    """Child processes started by the head, all stopped together."""

    def __init__(self) -> None:
        self.running: list[tuple[str, subprocess.Popen]] = []

    def start(self, name: str, cmd: list[str], *, cwd: Path = REPO, env=None) -> Path:
        LOGS.mkdir(parents=True, exist_ok=True)
        log = LOGS / f"{name}.log"
        print(f"starting {name}  (log: {log})")
        # Own process group, so stopping a child also stops whatever it spawned
        # (npx -> next -> next-server) instead of orphaning it on the port.
        group = (
            {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
            if IS_WINDOWS
            else {"start_new_session": True}
        )
        with log.open("w") as out:
            proc = subprocess.Popen(
                cmd, cwd=cwd, env=env, stdout=out, stderr=subprocess.STDOUT, **group
            )
        self.running.append((name, proc))
        return log

    def dead(self) -> str | None:
        """Name of the first child that has exited, if any."""
        for name, proc in self.running:
            if proc.poll() is not None:
                return name
        return None

    def stop_all(self) -> None:
        for _, proc in reversed(self.running):
            if proc.poll() is not None:
                continue
            if IS_WINDOWS:
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
            else:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    continue
        for _, proc in self.running:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


def _wait_for_line(procs: Processes, name: str, log: Path, needle: str) -> None:
    """Block until ``needle`` appears in a child's log, or fail if it dies first."""
    while needle not in log.read_text(errors="replace"):
        if procs.dead() == name:
            print(f"{name} exited during startup:")
            print("\n".join(log.read_text(errors="replace").splitlines()[-20:]))
            raise SystemExit(1)
        time.sleep(1)


def _http_ok(url: str, body: dict | None = None, timeout: float = 5) -> bool:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"} if data else {}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout):
            return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# roles


def cmd_check(args) -> int:
    from torrent_llm.runner import TopologyConfig

    config = TopologyConfig.from_file(args.config)
    print(f"{config.model_id}, {config.num_layers} layers over {len(config.nodes)} shards")
    all_up, _ = check_nodes(config)
    print("chain is complete" if all_up else "chain is NOT complete")
    return 0 if all_up else 1


def cmd_worker(args) -> int:
    from torrent_llm.runner import TopologyConfig

    config = TopologyConfig.from_file(args.config)
    if not 0 <= args.index < len(config.nodes):
        sys.exit(f"--index must be in [0, {len(config.nodes)}), got {args.index}")
    node = config.nodes[args.index]
    mine = local_addresses()

    print(f"this machine: {socket.gethostname()} ({platform.system()} {platform.machine()})")
    print(f"its addresses: {', '.join(mine) or 'none found'}")
    print(f"config says shard {args.index} is at {node.address}")
    if node.host not in mine and not node.host.startswith("127."):
        print(
            f"WARNING: {node.host} is not one of this machine's addresses. The head "
            "will dial that address, so it will not reach this shard. Fix the config "
            "and copy it to every machine."
        )
    print()
    print(firewall_hint(node.port))
    print()

    cmd = [
        _console_script("torrent-shard"),
        "--config",
        args.config,
        "--index",
        str(args.index),
        "--bind",
        args.bind,
    ]
    if args.device:
        cmd += ["--device", args.device]
    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        return 0


def cmd_head(args) -> int:
    from torrent_llm.runner import TopologyConfig

    config = TopologyConfig.from_file(args.config)
    local = (
        args.local
        if args.local is not None
        else [
            i
            for i, n in enumerate(config.nodes)
            if n.host.startswith("127.") or n.host == "localhost"
        ]
    )
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    procs = Processes()
    _stop_on_signals()

    try:
        for i in local:
            host = config.nodes[i].host
            # Loopback-only topologies keep their loopback binding; anything
            # else listens everywhere so the address in the config resolves.
            bind = host if host.startswith("127.") else "0.0.0.0"
            cmd = [
                _console_script("torrent-shard"),
                "--config",
                args.config,
                "--index",
                str(i),
                "--bind",
                bind,
            ]
            log = procs.start(f"shard-{i}", cmd, env=env)
            _wait_for_line(procs, f"shard-{i}", log, "serving on")
            print(f"shard {i} is up")

        print("waiting for every shard in the chain to answer")
        while True:
            all_up, fatal = check_nodes(config, quiet=True)
            if fatal:
                return 1
            if all_up:
                break
            if procs.dead():
                print(f"{procs.dead()} exited; see its log in {LOGS}")
                return 1
            time.sleep(3)

        api = f"http://127.0.0.1:{args.api_port}"
        web = f"http://localhost:{args.web_port}"
        procs.start(
            "api",
            [
                _console_script("torrent-api"),
                "--config",
                args.config,
                "--port",
                str(args.api_port),
                "--cors-origin",
                web,
                "--cors-origin",
                f"http://127.0.0.1:{args.web_port}",
            ],
            env=env,
        )
        if not args.no_dashboard:
            _start_dashboard(procs, args.web_port, api, env)

        while not _http_ok(f"{api}/health"):
            if procs.dead():
                print(f"{procs.dead()} exited; see its log in {LOGS}")
                return 1
            time.sleep(1)
        if not args.no_dashboard:
            while not _http_ok(web):
                time.sleep(1)

        # The first request pays for kernel setup and the first connection to
        # every peer. Spend that here, not in front of the audience.
        print("warming up the model")
        _http_ok(
            f"{api}/generate",
            {"prompt": "Hi", "max_new_tokens": 2, "chat": True},
            timeout=300,
        )

        print()
        check_nodes(config)
        print()
        print(f"ready: open {web if not args.no_dashboard else api + '/docs'}")
        print("Ctrl+C to stop everything")
        while not procs.dead():
            time.sleep(1)
        print(f"{procs.dead()} exited; see its log in {LOGS}")
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        print("\nstopping")
        procs.stop_all()


def _stop_on_signals() -> None:
    """Make every way of ending the head clean up like Ctrl+C does.

    The children run in their own process groups, so a closed terminal (SIGHUP)
    or a ``kill`` (SIGTERM) would otherwise end only the head and leave the
    shards and the dashboard running, still holding their ports. SIGINT is
    reset too, because a shell starts background jobs with it ignored.
    """

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, signal.default_int_handler)
    for name in ("SIGTERM", "SIGHUP", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), interrupt)


def _start_dashboard(procs: Processes, port: int, api: str, env: dict) -> None:
    npx, npm = shutil.which("npx"), shutil.which("npm")
    if npx is None or npm is None:
        sys.exit("Node.js (npm/npx) is needed for the dashboard; install it or pass --no-dashboard")
    dashboard = REPO / "dashboard"
    if not (dashboard / "node_modules").is_dir():
        print("installing dashboard dependencies (first run on this checkout)")
        subprocess.run([npm, "ci", "--no-audit", "--no-fund"], cwd=dashboard, check=True)
    procs.start(
        "web",
        [npx, "next", "dev", "--port", str(port)],
        cwd=dashboard,
        env={**env, "NEXT_PUBLIC_API_BASE": api},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="role", required=True)

    check = sub.add_parser("check", help="ask every node what it hosts")
    check.add_argument("--config", required=True)
    check.set_defaults(run=cmd_check)

    worker = sub.add_parser("worker", help="host one shard on this machine")
    worker.add_argument("--config", required=True)
    worker.add_argument("--index", type=int, required=True)
    worker.add_argument("--bind", default="0.0.0.0", help="listen address (default: all)")
    worker.add_argument("--device", default=None, help="override: cpu, cuda, mps or auto")
    worker.set_defaults(run=cmd_worker)

    head = sub.add_parser("head", help="host local shards, then the API and dashboard")
    head.add_argument("--config", required=True)
    head.add_argument(
        "--local",
        type=int,
        nargs="*",
        default=None,
        help="shard indexes to start on this machine (default: the loopback ones)",
    )
    head.add_argument("--api-port", type=int, default=8000)
    head.add_argument("--web-port", type=int, default=3000)
    head.add_argument("--no-dashboard", action="store_true")
    head.set_defaults(run=cmd_head)

    args = parser.parse_args()
    _reexec_in_venv()
    # Progress lines must show up as they happen, even when piped to a file.
    sys.stdout.reconfigure(line_buffering=True)
    # Resolved before the chdir, so a relative path means relative to where the
    # command was typed, as anyone would expect.
    args.config = str(Path(args.config).resolve())
    os.chdir(REPO)
    _ensure_stubs()
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
