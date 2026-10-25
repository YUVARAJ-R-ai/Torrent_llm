"""Noticing that the owner wants their GPU: parsing, and when to back off and return."""

from torrent_llm.pool.owner import GpuProcess, OwnerMonitor, owner_load, parse_pmon

# Real output from `nvidia-smi pmon -c 1 -s um`, trimmed.
PMON = """\
# gpu         pid   type     sm    mem    enc    dec    jpg    ofa     fb   ccpm    command
# Idx           #    C/G      %      %      %      %      %      %     MB     MB    name
    0        797     G      -      -      -      -      -      -     15      0    Xorg
    0       1254     G      -      -      -      -      -      -    105      0    Hyprland
    0       1628     G     12      3      -      -      -      -    209      0    firefox
    0     269658     C     71     40      -      -      -      -   3462      0    python3
"""


def proc(pid, mb, sm=0.0):
    return GpuProcess(pid=pid, kind="G", sm_percent=sm, memory_mb=mb, command="x")


def test_pmon_output_is_parsed_with_missing_columns_as_zero():
    processes = parse_pmon(PMON)

    assert [p.pid for p in processes] == [797, 1254, 1628, 269658]
    assert processes[0].sm_percent == 0.0 and processes[0].memory_mb == 15
    assert processes[2].sm_percent == 12.0
    assert processes[3].memory_mb == 3462 and processes[3].kind == "C"


def test_headers_and_blank_lines_are_ignored():
    assert parse_pmon("# header\n\n") == []


def test_the_pools_own_processes_do_not_count_as_the_owner():
    processes = [proc(1, 3000), proc(2, 400)]

    mb, _ = owner_load(processes, is_pool=lambda pid: pid == 1)

    assert mb == 400


def test_the_busiest_other_process_sets_the_compute_share():
    processes = [proc(1, 10, sm=80.0), proc(2, 10, sm=20.0)]

    _, sm = owner_load(processes, is_pool=lambda pid: pid == 1)

    assert sm == 20.0


class Script:
    """A probe that plays back a list of readings, then repeats the last."""

    def __init__(self, readings):
        self._readings = list(readings)

    def __call__(self):
        value = self._readings.pop(0) if len(self._readings) > 1 else self._readings[0]
        return None if value is None else [proc(9, value)]


def monitor(readings, **kw):
    calls = []
    m = OwnerMonitor(
        calls.append,
        probe=Script(readings),
        is_pool=lambda pid: False,
        busy_mb=1000,
        busy_after=2,
        idle_after=3,
        **kw,
    )
    return m, calls


def run(m, n):
    for _ in range(n):
        m.sample()


def test_a_desktop_and_browser_are_not_the_owner_using_the_gpu():
    m, calls = monitor([400])

    run(m, 20)

    assert calls == []


def test_it_backs_off_only_after_the_gpu_stays_taken():
    m, calls = monitor([2000, 100, 2000, 2000])  # one blip, then a real game

    m.sample()
    m.sample()  # the blip is over
    assert calls == []
    m.sample()
    m.sample()

    assert calls == [True]


def test_it_comes_back_only_after_the_owner_has_really_stopped():
    m, calls = monitor([2000, 2000, 100, 100, 2000, 100, 100, 100])

    run(m, 2)
    assert calls == [True]
    run(m, 2)  # two quiet readings, then the menu flickers back
    run(m, 4)

    assert calls == [True, False]


def test_it_never_undoes_what_the_owner_did_by_hand():
    m, calls = monitor([100])

    run(m, 20)  # the GPU is idle all along, but the monitor never backed off

    assert calls == []


def test_a_failing_probe_is_ignored():
    m, calls = monitor([None])

    run(m, 5)

    assert calls == []


def test_a_busy_compute_share_alone_counts_as_use():
    calls = []
    m = OwnerMonitor(
        calls.append,
        probe=lambda: [proc(9, 50, sm=90.0)],
        is_pool=lambda pid: False,
        busy_after=1,
    )

    m.sample()

    assert calls == [True]
