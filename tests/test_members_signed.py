"""Signed member entries: nobody can forge or alter another device's row."""

from __future__ import annotations

import dataclasses

import pytest

from torrent_llm.net.identity import Identity
from torrent_llm.pool.members import HostedStage, Member, MemberTable


def _row(identity: Identity, heartbeat: int = 1, **kw) -> Member:
    return Member(
        name=identity.peer_id,
        host="",
        control_port=0,
        label="alice",
        heartbeat=heartbeat,
        hosting=(HostedStage(stage=0, status="ready"),),
        addrs=("tcp://10.0.0.2:9100",),
        **kw,
    ).signed_by(identity)


def test_a_signed_entry_verifies_and_survives_the_wire():
    me = Identity.generate()
    row = _row(me)
    assert row.verify()
    again = Member.from_dict(row.to_dict())
    assert again == row and again.verify()


def test_any_change_or_a_different_key_breaks_the_signature():
    me, other = Identity.generate(), Identity.generate()
    row = _row(me)
    assert not dataclasses.replace(row, label="mallory").verify()
    assert not dataclasses.replace(row, heartbeat=99).verify()
    assert not dataclasses.replace(row, addrs=("tcp://6.6.6.6:1",)).verify()
    forged = dataclasses.replace(_row(other), name=me.peer_id)
    assert not forged.verify()
    with pytest.raises(ValueError):
        Member(name=me.peer_id, host="", control_port=0).signed_by(other)


def test_a_table_that_requires_signatures_drops_forgeries():
    me, alice, mallory = Identity.generate(), Identity.generate(), Identity.generate()
    table = MemberTable(me.peer_id, require_signatures=True)
    good = _row(alice)
    forged = dataclasses.replace(_row(mallory, heartbeat=50), name=alice.peer_id)
    assert table.merge([good]) == [alice.peer_id]
    assert table.merge([forged]) == []  # newer stamp, but not signed by alice
    assert table.get(alice.peer_id) == good
    unsigned = Member(name=Identity.generate().peer_id, host="", control_port=0, heartbeat=1)
    assert table.merge([unsigned]) == []


def test_digest_gossip_sends_only_what_the_other_side_lacks():
    me, a, b = Identity.generate(), Identity.generate(), Identity.generate()
    table = MemberTable(me.peer_id)
    table.merge([_row(a, 5), _row(b, 1)])
    theirs = {a.peer_id: [0, 5], b.peer_id: [0, 0]}
    assert [m.name for m in table.newer_than(theirs)] == [b.peer_id]
    assert {m.name for m in table.newer_than({})} == {a.peer_id, b.peer_id}
    assert table.stamps()[a.peer_id] == [0, 5]


def test_untrusted_entries_are_refused_with_value_error():
    for raw in (
        None,
        [],
        {"name": "x", "host": "", "control_port": 0, "evil": 1},
        {"name": "x", "host": "", "control_port": 0, "hosting": [{"stage": "a", "x": 1}]},
        {"name": "x", "host": "", "control_port": 0, "addrs": ["a"] * 17},
    ):
        with pytest.raises(ValueError):
            Member.from_dict(raw)


def test_the_table_is_bounded_and_forgets_long_gone_devices():
    now = [0.0]
    table = MemberTable("me", max_members=3, clock=lambda: now[0])
    rows = [Member(name=f"d{i}", host="h", control_port=i, heartbeat=1) for i in range(5)]
    table.merge(rows)
    assert len(table.members()) == 3
    now[0] = 100.0
    assert len(table.forget(older_than_s=50)) == 3
    assert table.members() == []
