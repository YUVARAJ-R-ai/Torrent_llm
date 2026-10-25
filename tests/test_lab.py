"""The lab that plays pretend laptops: starting, stopping and crashing processes.

Uses ``sleep`` as the stand-in program, so no model is loaded and nothing here
depends on the pool itself.
"""

import pytest
from fastapi.testclient import TestClient

from torrent_llm.pool.config import DeviceProfile, LabConfig, PoolConfig
from torrent_llm.pool.lab import BASE_PORT, Lab, create_lab_app, peer_command


@pytest.fixture
def lab(tmp_path):
    config = LabConfig(
        pool=PoolConfig(model_id="tiny", num_layers=6),
        devices=[DeviceProfile(name="a"), DeviceProfile(name="b")],
    )
    seen: list[list[str]] = []

    def command(device, join):
        seen.append(list(join))
        return ["sleep", "60"]

    lab = Lab(config, command, log_dir=tmp_path)
    lab.joins = seen
    yield lab
    lab.shutdown()


def test_devices_get_their_own_ports(lab):
    assert [d.port for d in lab.devices.values()] == [BASE_PORT, BASE_PORT + 1]


def test_a_device_starts_and_is_reported_running(lab):
    lab.start("a")

    assert lab.devices["a"].running
    assert not lab.devices["b"].running


def test_starting_a_running_device_does_nothing(lab):
    first = lab.start("a").process

    assert lab.start("a").process is first


def test_a_later_device_is_told_where_the_running_ones_are(lab):
    lab.start("a")
    lab.start("b")

    assert lab.joins == [[], [f"127.0.0.1:{BASE_PORT}"]]


def test_a_killed_device_is_dead_and_says_it_crashed(lab):
    lab.start("a")

    lab.kill("a")

    assert not lab.devices["a"].running
    assert lab.devices["a"].last_event == "crashed"


def test_a_stopped_device_says_it_was_switched_off(lab):
    lab.start("a")

    lab.stop("a")

    assert not lab.devices["a"].running
    assert lab.devices["a"].last_event == "switched off"


def test_a_device_can_be_brought_back_after_a_crash(lab):
    lab.start("a")
    lab.kill("a")

    lab.start("a")

    assert lab.devices["a"].running


def test_the_peer_command_names_the_profile_and_port(tmp_path):
    config = LabConfig(
        pool=PoolConfig(model_id="tiny", num_layers=6),
        devices=[DeviceProfile(name="gaming")],
    )
    lab = Lab(config, peer_command(tmp_path / "pool.yaml", "tok", ["http://x"]), log_dir=tmp_path)

    cmd = lab._command(lab.devices["gaming"], ["127.0.0.1:9101"])

    assert cmd[cmd.index("--profile") + 1] == "gaming"
    assert cmd[cmd.index("--port") + 1] == str(BASE_PORT)
    assert cmd[cmd.index("--join") + 1] == "127.0.0.1:9101"
    assert cmd[cmd.index("--token") + 1] == "tok"
    assert cmd[cmd.index("--cors-origin") + 1] == "http://x"


def test_the_api_lists_starts_and_kills_devices(lab):
    client = TestClient(create_lab_app(lab))

    assert client.post("/lab/devices/a/start").status_code == 200
    state = client.get("/lab").json()
    assert [d["name"] for d in state["devices"]] == ["a", "b"]
    assert state["devices"][0]["running"] is True
    assert state["devices"][0]["pid"] is not None

    client.post("/lab/devices/a/kill")
    assert client.get("/lab").json()["devices"][0]["running"] is False


def test_the_api_names_an_unknown_device(lab):
    client = TestClient(create_lab_app(lab))

    response = client.post("/lab/devices/nope/kill")

    assert response.status_code == 404
    assert "nope" in response.json()["detail"]
