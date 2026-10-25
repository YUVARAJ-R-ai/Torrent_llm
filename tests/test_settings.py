"""What torrent-peer decides before it starts, and what the readiness check says."""

import pytest

from torrent_llm.pool import doctor, net
from torrent_llm.pool.settings import PRIVATE_ORIGIN_REGEX, build_parser, settings_from_args

POOL = "configs/pool.yaml"


@pytest.fixture(autouse=True)
def quiet_machine(monkeypatch):
    """A predictable machine: a laptop on 10.0.0.5 with 8000 MB of RAM and no GPU."""
    monkeypatch.setattr(net, "lan_address", lambda: "10.0.0.5")
    monkeypatch.setattr(net, "available_ram_mb", lambda: 8000)
    monkeypatch.setattr(net, "free_vram_mb", lambda: None)
    monkeypatch.setattr(
        "torrent_llm.pool.settings.resolve_device", lambda d: "cpu" if d == "auto" else d
    )


def settings(*argv):
    return settings_from_args(build_parser().parse_args([*argv]))


def test_with_no_options_it_works_everything_out_itself():
    s = settings("--pool", POOL)

    assert s.host == "10.0.0.5"
    assert s.bind == "0.0.0.0"
    assert s.profile.memory_mb == 3200  # 40% of 8000
    assert s.memory_source == "a share of what is free"
    assert s.discover is True
    assert s.port == 9000
    assert s.shard_ports == (9200, 9201)


def test_the_name_defaults_to_the_hostname(monkeypatch):
    monkeypatch.setattr("torrent_llm.pool.settings.socket.gethostname", lambda: "alices-laptop")

    assert settings("--pool", POOL).profile.name == "alices-laptop"


def test_an_explicit_name_and_memory_are_respected():
    s = settings("--pool", POOL, "--name", "bob", "--memory-mb", "2000")

    assert s.profile.name == "bob"
    assert s.profile.memory_mb == 2000
    assert s.memory_source == "set by you"


def test_a_loopback_host_listens_only_on_loopback():
    s = settings("--pool", POOL, "--host", "127.0.0.1")

    assert s.bind == "127.0.0.1"


def test_a_real_address_listens_everywhere_so_others_can_connect():
    assert settings("--pool", POOL, "--host", "192.168.1.9").bind == "0.0.0.0"


def test_no_network_is_explained_not_hidden(monkeypatch):
    monkeypatch.setattr(net, "lan_address", lambda: "127.0.0.1")

    s = settings("--pool", POOL)

    assert any("no network address" in n for n in s.notes)


def test_a_gpu_device_watches_its_owner_by_default(monkeypatch):
    monkeypatch.setattr("torrent_llm.pool.settings.resolve_device", lambda d: "cuda")
    monkeypatch.setattr(net, "free_vram_mb", lambda: 6000)

    s = settings("--pool", POOL)

    assert s.profile.watch_owner is True
    assert s.profile.memory_mb == 4200  # 70% of free VRAM


def test_a_cpu_device_does_not_watch_a_gpu_that_is_not_there():
    assert settings("--pool", POOL).profile.watch_owner is False


def test_watching_can_be_turned_off():
    monkeypatch_free = settings("--pool", POOL, "--no-watch-owner")

    assert monkeypatch_free.profile.watch_owner is False


def test_a_lab_profile_is_taken_from_the_pool_file():
    s = settings("--pool", "configs/pool-lab.yaml", "--profile", "macbook", "--host", "127.0.0.1")

    assert s.profile.name == "macbook"
    assert s.profile.memory_mb == 1600  # from the file, not measured
    assert s.memory_source == "set in the pool file"


def test_an_unknown_lab_profile_lists_the_choices():
    with pytest.raises(SystemExit, match="gaming"):
        settings("--pool", "configs/pool-lab.yaml", "--profile", "nope")


def test_a_missing_pool_file_says_where_to_run_from():
    with pytest.raises(SystemExit, match="pool file not found"):
        settings("--pool", "does-not-exist.yaml")


def test_pages_from_private_networks_may_call_a_device_but_the_open_internet_may_not():
    import re

    allowed = re.compile(PRIVATE_ORIGIN_REGEX)

    for origin in [
        "http://localhost:3000",
        "http://192.168.1.4:9000",
        "http://10.1.2.3:9000",
        "http://172.20.0.1",
        "http://100.101.2.3:9000",
    ]:
        assert allowed.match(origin), origin
    for origin in [
        "https://evil.example.com",
        "http://8.8.8.8",
        "http://192.169.1.1",
        "http://172.32.0.1",
        "http://100.200.1.1",
    ]:
        assert not allowed.match(origin), origin


def test_a_token_in_the_environment_is_picked_up(monkeypatch):
    monkeypatch.setenv("TORRENT_POOL_TOKEN", "from-env")

    assert settings("--pool", POOL).token == "from-env"


# --- the readiness check ------------------------------------------------------------------


def test_a_machine_with_too_little_memory_is_warned_it_will_hold_nothing():
    s = settings("--pool", POOL, "--memory-mb", "500")

    check = doctor.check_memory(s)

    assert check.status == doctor.WARN
    assert "hold nothing" in check.detail and "--memory-mb" in check.fix


def test_enough_memory_says_how_many_stages_fit():
    check = doctor.check_memory(settings("--pool", POOL, "--memory-mb", "3200"))

    assert check.status == doctor.OK and "2 stages" in check.detail


def test_a_busy_port_is_a_failure_that_names_it():
    import socket

    with socket.socket() as held:
        held.bind(("0.0.0.0", 0))
        held.listen()
        busy = held.getsockname()[1]

        check = doctor.check_ports(settings("--pool", POOL, "--port", str(busy)))

    assert check.status == doctor.FAIL and str(busy) in check.detail


def test_no_network_address_is_a_warning_with_a_fix(monkeypatch):
    monkeypatch.setattr(net, "lan_address", lambda: "127.0.0.1")

    check = doctor.check_network(settings("--pool", POOL))

    assert check.status == doctor.WARN and "--host" in check.fix


def test_a_missing_model_is_a_failure_that_says_how_to_get_it(monkeypatch):
    import huggingface_hub

    def not_cached(*args, **kwargs):
        raise FileNotFoundError("not cached")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", not_cached)

    check = doctor.check_model(settings("--pool", POOL))

    assert check.status == doctor.FAIL and "setup" in check.fix


def test_an_unreachable_seed_is_a_failure_that_names_it():
    checks = doctor.check_seeds(settings("--pool", POOL, "--join", "127.0.0.1:1"))

    assert checks[0].status == doctor.FAIL and "127.0.0.1:1" in checks[0].name


def test_the_report_counts_problems_and_says_when_ready():
    ok = doctor.Check("a", doctor.OK, "fine")
    bad = doctor.Check("b", doctor.FAIL, "broken", "do this")

    assert "Ready to join." in doctor.report([ok])
    text = doctor.report([ok, bad])
    assert "1 problem to fix" in text and "do this" in text
