"""Working out this machine's address and memory, and finding other devices."""

import ipaddress
import socket
import threading
import time

import pytest

from torrent_llm.pool import net
from torrent_llm.pool.discovery import MAGIC, Announcement, Discovery


def free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return False


# --- this machine ---------------------------------------------------------------


def test_the_lan_address_is_a_valid_ipv4_address():
    assert isinstance(ipaddress.ip_address(net.lan_address()), ipaddress.IPv4Address)


def test_an_explicit_host_is_used_as_given():
    assert net.choose_host("192.168.1.20") == "192.168.1.20"


def test_auto_and_lan_both_mean_the_lan_address(monkeypatch):
    monkeypatch.setattr(net, "lan_address", lambda: "10.0.0.5")

    assert net.choose_host("auto") == net.choose_host("lan") == "10.0.0.5"


def test_asking_for_tailscale_without_it_says_how_to_get_it(monkeypatch):
    monkeypatch.setattr(net, "tailscale_address", lambda: None)

    with pytest.raises(RuntimeError, match="tailscale.com"):
        net.choose_host("tailscale")


def test_a_gpu_lends_most_of_its_free_memory(monkeypatch):
    monkeypatch.setattr(net, "free_vram_mb", lambda: 6000)
    monkeypatch.setattr(net, "available_ram_mb", lambda: 16000)

    assert net.auto_memory_mb("cuda") == 4200


def test_a_cpu_lends_a_smaller_share_of_ram(monkeypatch):
    monkeypatch.setattr(net, "available_ram_mb", lambda: 10000)

    assert net.auto_memory_mb("cpu") == 4000


def test_a_cuda_device_with_no_readable_gpu_falls_back_to_ram(monkeypatch):
    monkeypatch.setattr(net, "free_vram_mb", lambda: None)
    monkeypatch.setattr(net, "available_ram_mb", lambda: 10000)

    assert net.auto_memory_mb("cuda") == 4000


def test_a_port_in_use_is_reported_as_not_free():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
        held.bind(("0.0.0.0", 0))
        held.listen()
        port = held.getsockname()[1]

        assert not net.port_is_free(port)

    assert net.port_is_free(port)


@pytest.mark.parametrize(
    ("system", "needle"),
    [("Windows", "New-NetFirewallRule"), ("Darwin", "Allow"), ("Linux", "9099")],
)
def test_firewall_advice_is_for_this_operating_system(monkeypatch, system, needle):
    monkeypatch.setattr(net.platform, "system", lambda: system)

    advice = net.firewall_advice(9000, (9200, 9210))

    assert needle in advice


def test_firewall_advice_names_every_port_that_must_be_open(monkeypatch):
    monkeypatch.setattr(net.platform, "system", lambda: "Windows")

    advice = net.firewall_advice(9000, (9200, 9210))

    assert "9000" in advice and "9200-9210" in advice


# --- announcements ---------------------------------------------------------------


def test_an_announcement_survives_the_wire():
    a = Announcement(name="laptop", host="192.168.1.5", port=9000, pool="m/2/2")

    assert Announcement.decode(a.encode()) == a
    assert a.address == "192.168.1.5:9000"


@pytest.mark.parametrize(
    "junk",
    [
        b"",
        b"hello",
        b"\xff\xfe",
        b"{}",
        b'{"magic": "something-else"}',
        b'{"magic": "%s"}' % MAGIC.encode(),
        b"[1, 2]",
    ],
)
def test_anything_that_is_not_ours_is_ignored(junk):
    assert Announcement.decode(junk) is None


# --- finding each other ------------------------------------------------------------


def pair(pool_a="m", pool_b="m"):
    """Two devices that can hear each other over loopback, and what each heard."""
    pa, pb = free_udp_port(), free_udp_port()
    heard_a, heard_b = [], []
    a = Discovery(
        Announcement("a", "127.0.0.1", 9001, pool_a),
        heard_a.append,
        port=pa,
        targets=[("127.0.0.1", pb)],
        interval_s=0.05,
    )
    b = Discovery(
        Announcement("b", "127.0.0.1", 9002, pool_b),
        heard_b.append,
        port=pb,
        targets=[("127.0.0.1", pa)],
        interval_s=0.05,
    )
    return a, b, heard_a, heard_b


def test_two_devices_find_each_other():
    a, b, heard_a, heard_b = pair()
    a.start()
    b.start()
    try:
        assert wait_for(lambda: heard_a and heard_b)
    finally:
        a.stop()
        b.stop()

    assert heard_a[0].address == "127.0.0.1:9002" and heard_a[0].name == "b"
    assert heard_b[0].address == "127.0.0.1:9001" and heard_b[0].name == "a"


def test_a_device_does_not_report_itself():
    port = free_udp_port()
    heard = []
    me = Announcement("solo", "127.0.0.1", 9001, "m")
    d = Discovery(me, heard.append, port=port, targets=[("127.0.0.1", port)], interval_s=0.05)
    d.start()
    try:
        time.sleep(0.4)
    finally:
        d.stop()

    assert heard == []


def test_a_device_from_another_pool_is_ignored():
    a, b, heard_a, heard_b = pair(pool_a="one", pool_b="two")
    a.start()
    b.start()
    try:
        time.sleep(0.5)
    finally:
        a.stop()
        b.stop()

    assert heard_a == [] and heard_b == []


def test_a_failing_callback_does_not_stop_discovery():
    pa, pb = free_udp_port(), free_udp_port()
    calls = []

    def flaky(found):
        calls.append(found)
        raise RuntimeError("boom")

    a = Discovery(
        Announcement("a", "127.0.0.1", 1, "m"),
        flaky,
        port=pa,
        targets=[("127.0.0.1", pb)],
        interval_s=0.05,
    )
    b = Discovery(
        Announcement("b", "127.0.0.1", 2, "m"),
        lambda f: None,
        port=pb,
        targets=[("127.0.0.1", pa)],
        interval_s=0.05,
    )
    a.start()
    b.start()
    try:
        assert wait_for(lambda: len(calls) >= 3)
    finally:
        a.stop()
        b.stop()


def test_stopping_ends_its_threads():
    a, b, *_ = pair()
    a.start()
    b.start()
    a.stop()
    b.stop()

    assert not any(t.is_alive() for t in a._threads + b._threads)


def test_a_real_broadcast_reaches_a_listener_on_this_machine():
    # The one test that uses the real broadcast address. Skipped where the sandbox
    # or network does not allow broadcasting, since that is not what is under test.
    port = free_udp_port()
    heard = threading.Event()
    listener = Discovery(
        Announcement("l", "127.0.0.1", 1, "m"), lambda f: heard.set(), port=port, interval_s=30
    )
    sender = Discovery(
        Announcement("s", "127.0.0.1", 2, "m"),
        lambda f: None,
        port=port,
        targets=[("255.255.255.255", port)],
        interval_s=0.1,
    )
    listener.start()
    sender.start()
    try:
        got = heard.wait(3.0)
        if not got and sender.send_error:
            pytest.skip(f"broadcast not allowed here: {sender.send_error}")
        assert got
    finally:
        listener.stop()
        sender.stop()
