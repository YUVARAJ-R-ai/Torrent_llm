"""Tests for signed peer records and address parsing."""

from __future__ import annotations

import dataclasses
import time

import pytest

from torrent_llm.net.identity import Identity
from torrent_llm.net.records import (
    MAX_ADDRESSES,
    MAX_CLOCK_SKEW_S,
    PeerRecord,
    RelayAddress,
    TcpAddress,
    make_record,
    parse_address,
)


def test_parse_address_docstring_examples() -> None:
    peer_id = Identity.generate().peer_id
    assert parse_address("tcp://1.2.3.4:9100") == TcpAddress("1.2.3.4", 9100)
    assert parse_address("tcp://[::1]:80") == TcpAddress("::1", 80)
    assert parse_address("tcp://host.example:9") == TcpAddress("host.example", 9)
    assert parse_address("relay://" + peer_id) == RelayAddress(peer_id)

    with pytest.raises(ValueError, match="port 1-65535|port must be 1-65535"):
        parse_address("tcp://1.2.3.4:0")

    with pytest.raises(ValueError, match="no port"):
        parse_address("tcp://1.2.3.4")

    with pytest.raises(ValueError, match="unknown address scheme"):
        parse_address("udp://1.2.3.4:5")

    with pytest.raises(ValueError, match="not a peer id"):
        parse_address("relay://not-an-id")

    with pytest.raises(ValueError, match="invalid host"):
        parse_address("tcp://bad host:5")


def test_parse_address_edge_cases() -> None:
    with pytest.raises(ValueError, match="address must be str"):
        parse_address(123)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="unknown address scheme"):
        parse_address("garbage-address")

    with pytest.raises(ValueError, match="empty host"):
        parse_address("tcp://:80")

    with pytest.raises(ValueError, match="unclosed IPv6 bracket"):
        parse_address("tcp://[::1:80")

    with pytest.raises(ValueError, match="no port"):
        parse_address("tcp://[::1]")

    with pytest.raises(ValueError, match="invalid IPv6 host"):
        parse_address("tcp://[not-an-ipv6]:80")

    with pytest.raises(ValueError, match="invalid address or unbracketed IPv6"):
        parse_address("tcp://::1:80")

    with pytest.raises(ValueError, match="invalid IPv4 host"):
        parse_address("tcp://999.999.999.999:80")

    with pytest.raises(ValueError, match="port must be 1-65535"):
        parse_address("tcp://1.2.3.4:0080")

    with pytest.raises(ValueError, match="port must be 1-65535"):
        parse_address("tcp://1.2.3.4:" + "9" * 100)

    with pytest.raises(ValueError, match="port must be 1-65535"):
        parse_address("tcp://1.2.3.4:65536")

    with pytest.raises(ValueError, match="port must be 1-65535"):
        parse_address("tcp://1.2.3.4:+80")

    with pytest.raises(ValueError, match="port must be 1-65535"):
        parse_address("tcp://1.2.3.4:-1")

    # Round trip string tests
    peer_id = Identity.generate().peer_id
    for raw in [
        "tcp://192.168.1.20:9100",
        "tcp://[2001:db8::1]:9100",
        "tcp://example.org:9100",
        f"relay://{peer_id}",
    ]:
        assert str(parse_address(raw)) == raw


def test_tcp_address_is_private() -> None:
    assert TcpAddress("10.0.0.1", 80).is_private is True
    assert TcpAddress("192.168.1.1", 80).is_private is True
    assert TcpAddress("127.0.0.1", 80).is_private is True
    assert TcpAddress("::1", 80).is_private is True
    assert TcpAddress("fe80::1", 80).is_private is True
    assert TcpAddress("fd00::1", 80).is_private is True
    assert TcpAddress("8.8.8.8", 80).is_private is False
    assert TcpAddress("example.org", 80).is_private is False
    assert TcpAddress("localhost", 80).is_private is False


def test_make_record_and_verify_round_trip() -> None:
    identity = Identity.generate()
    addresses = ["tcp://127.0.0.1:9000", "tcp://[::1]:9000"]
    record = make_record(
        identity,
        addresses,
        seq=1,
        attrs={"role": "worker", "active": True},
    )
    assert record.verify() is True
    assert len(record.parsed_addresses()) == 2
    assert record.peer_id == identity.peer_id
    assert record.public_key == identity.public_key
    assert record.seq == 1
    assert record.attrs == {"role": "worker", "active": True}


def test_make_record_address_dedup_and_order() -> None:
    identity = Identity.generate()
    addresses = ["tcp://1.1.1.1:80", "tcp://2.2.2.2:80", "tcp://1.1.1.1:80"]
    record = make_record(identity, addresses, seq=0)
    assert record.addresses == ("tcp://1.1.1.1:80", "tcp://2.2.2.2:80")


def test_make_record_limits_and_validation() -> None:
    identity = Identity.generate()

    with pytest.raises(ValueError, match="too many addresses"):
        make_record(
            identity,
            [f"tcp://1.1.1.{i}:80" for i in range(MAX_ADDRESSES + 1)],
            seq=0,
        )

    with pytest.raises(ValueError, match="seq must be non-negative"):
        make_record(identity, ["tcp://1.1.1.1:80"], seq=-1)

    with pytest.raises(ValueError, match="seq must be non-negative"):
        make_record(identity, ["tcp://1.1.1.1:80"], seq=True)  # bool is not int

    with pytest.raises(ValueError):
        make_record(identity, ["not-a-valid-addr"], seq=0)

    with pytest.raises(ValueError, match="invalid attrs"):
        make_record(
            identity,
            ["tcp://1.1.1.1:80"],
            seq=0,
            attrs={f"k{i}": i for i in range(17)},
        )

    with pytest.raises(ValueError, match="invalid attrs"):
        make_record(
            identity,
            ["tcp://1.1.1.1:80"],
            seq=0,
            attrs={"k" * 33: "val"},
        )

    with pytest.raises(ValueError, match="invalid attrs"):
        make_record(
            identity,
            ["tcp://1.1.1.1:80"],
            seq=0,
            attrs={"k": "v" * 257},
        )

    with pytest.raises(ValueError, match="invalid attrs"):
        make_record(
            identity,
            ["tcp://1.1.1.1:80"],
            seq=0,
            attrs={"k": [1, 2, 3]},
        )


def test_to_dict_from_dict_round_trip() -> None:
    identity = Identity.generate()
    record = make_record(
        identity,
        ["tcp://127.0.0.1:9000"],
        seq=5,
        attrs={"key": "val", "count": 10, "flag": False, "none": None},
    )
    d = record.to_dict()
    assert isinstance(d, dict)
    r2 = PeerRecord.from_dict(d)
    assert r2 == record
    assert r2.verify() is True


def test_from_dict_garbage_inputs() -> None:
    with pytest.raises(ValueError, match="expected dict"):
        PeerRecord.from_dict(None)

    with pytest.raises(ValueError, match="expected dict"):
        PeerRecord.from_dict([])

    with pytest.raises(ValueError, match="missing keys"):
        PeerRecord.from_dict({"peer_id": 1})

    identity = Identity.generate()
    valid_d = make_record(identity, ["tcp://1.1.1.1:80"], seq=0).to_dict()

    bad_field_cases = [
        ("peer_id", 123),
        ("public_key", 123),
        ("public_key", "bad-base64-!@#$"),
        ("addresses", "not-a-list"),
        ("addresses", [123]),
        ("seq", True),
        ("seq", "1"),
        ("issued_at", "now"),
        ("attrs", "not-a-dict"),
        ("signature", 123),
        ("signature", "bad-base64-!@#$"),
    ]
    for key, val in bad_field_cases:
        bad_d = dict(valid_d)
        bad_d[key] = val
        with pytest.raises(ValueError):
            PeerRecord.from_dict(bad_d)


def test_verify_tampering_individual_fields() -> None:
    identity = Identity.generate()
    other_identity = Identity.generate()
    valid_record = make_record(
        identity,
        ["tcp://127.0.0.1:9000"],
        seq=2,
        attrs={"role": "peer", "ok": True},
    )
    assert valid_record.verify() is True
    assert dataclasses.replace(valid_record, addresses=("not-an-address",)).verify() is False
    assert dataclasses.replace(valid_record, seq=True).verify() is False
    assert dataclasses.replace(valid_record, seq=-1).verify() is False
    assert dataclasses.replace(valid_record, seq=3).verify() is False
    assert dataclasses.replace(valid_record, public_key=b"short").verify() is False
    assert dataclasses.replace(valid_record, public_key="not-bytes").verify() is False
    assert dataclasses.replace(valid_record, public_key=other_identity.public_key).verify() is False
    assert (
        dataclasses.replace(
            valid_record,
            public_key=other_identity.public_key,
            peer_id=other_identity.peer_id,
        ).verify()
        is False
    )
    assert dataclasses.replace(valid_record, peer_id=other_identity.peer_id).verify() is False
    assert dataclasses.replace(valid_record, peer_id="bad_id").verify() is False
    assert dataclasses.replace(valid_record, signature=b"x" * 64).verify() is False
    assert dataclasses.replace(valid_record, signature=b"short").verify() is False
    assert dataclasses.replace(valid_record, signature="not-bytes").verify() is False
    assert dataclasses.replace(valid_record, addresses=("tcp://127.0.0.1:9001",)).verify() is False
    assert (
        dataclasses.replace(
            valid_record,
            addresses=tuple(f"tcp://1.1.1.{i}:80" for i in range(MAX_ADDRESSES + 1)),
        ).verify()
        is False
    )
    assert dataclasses.replace(valid_record, addresses="not-a-tuple").verify() is False
    assert dataclasses.replace(valid_record, attrs=None).verify() is False
    assert dataclasses.replace(valid_record, attrs={"role": "tampered"}).verify() is False
    assert dataclasses.replace(valid_record, attrs={"toolong": "a" * 300}).verify() is False
    assert dataclasses.replace(valid_record, attrs={"k" * 33: "v"}).verify() is False
    assert (
        dataclasses.replace(
            valid_record,
            attrs={f"k{i}": i for i in range(17)},
        ).verify()
        is False
    )
    assert dataclasses.replace(valid_record, attrs={"bad_val": [1, 2]}).verify() is False
    assert dataclasses.replace(valid_record, issued_at=valid_record.issued_at + 1).verify() is False
    assert dataclasses.replace(valid_record, issued_at="future").verify() is False
    now = time.time()
    future_rec = make_record(
        identity,
        ["tcp://127.0.0.1:9000"],
        seq=3,
        now=now + MAX_CLOCK_SKEW_S + 10,
    )
    assert future_rec.verify(now=now) is False

    valid_skew_rec = make_record(
        identity,
        ["tcp://127.0.0.1:9000"],
        seq=3,
        now=now + MAX_CLOCK_SKEW_S - 10,
    )
    assert valid_skew_rec.verify(now=now) is True


def test_newer_than_rules() -> None:
    id1 = Identity.generate()
    id2 = Identity.generate()

    r1_seq1 = make_record(id1, ["tcp://1.1.1.1:80"], seq=1)
    r1_seq2 = make_record(id1, ["tcp://1.1.1.1:80"], seq=2)
    r2_seq3 = make_record(id2, ["tcp://1.1.1.1:80"], seq=3)

    assert r1_seq1.newer_than(None) is True
    assert r1_seq1.newer_than("not a record") is False  # type: ignore[arg-type]
    assert r1_seq2.newer_than(r1_seq1) is True
    assert r1_seq1.newer_than(r1_seq2) is False
    assert r1_seq1.newer_than(r1_seq1) is False
    assert r1_seq2.newer_than(r2_seq3) is False
    assert r2_seq3.newer_than(r1_seq2) is False


def test_parsed_addresses() -> None:
    identity = Identity.generate()
    record = make_record(identity, ["tcp://1.1.1.1:80", "tcp://[::1]:80"], seq=0)
    parsed = record.parsed_addresses()
    assert len(parsed) == 2
    assert parsed[0] == TcpAddress("1.1.1.1", 80)
    assert parsed[1] == TcpAddress("::1", 80)

    # Skips unparseable address
    r_bad = dataclasses.replace(
        record,
        addresses=("tcp://1.1.1.1:80", "not-valid-address"),
    )
    assert r_bad.parsed_addresses() == [TcpAddress("1.1.1.1", 80)]

    # Non-iterable addresses returns empty list
    r_bad2 = dataclasses.replace(record, addresses=123)
    assert r_bad2.parsed_addresses() == []


def test_issued_at_is_whole_seconds_and_survives_a_json_round_trip():
    import json as _json

    from torrent_llm.net.identity import Identity as _Identity
    from torrent_llm.net.records import PeerRecord as _PeerRecord
    from torrent_llm.net.records import make_record as _make_record

    record = _make_record(_Identity.generate(), ["tcp://1.2.3.4:9100"], seq=1, now=1700000000.75)
    assert record.issued_at == 1700000000 and type(record.issued_at) is int
    again = _PeerRecord.from_dict(_json.loads(_json.dumps(record.to_dict())))
    assert again.verify(now=1700000001)
    for bad in (1700000000.0, 10**400, True, -1, float("nan")):
        raw = {**record.to_dict(), "issued_at": bad}
        with pytest.raises(ValueError):
            _PeerRecord.from_dict(raw)
    assert not record.verify(now=float("nan"))


@pytest.mark.parametrize(
    "text",
    [
        "tcp://example.org\n:80",
        "tcp://1.2.3.4:\u0668\u0660",
        "tcp://1.2.3.4:\u00b2",
        "tcp://ex\u00e4mple.org:80",
    ],
)
def test_addresses_must_be_plain_ascii(text):
    from torrent_llm.net.records import parse_address as _parse

    with pytest.raises(ValueError):
        _parse(text)
