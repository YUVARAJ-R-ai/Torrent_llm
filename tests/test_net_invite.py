"""Tests for invite links: generation, encoding, parsing, validation, and expiry."""

from __future__ import annotations

import base64
import json
import re

import pytest

import torrent_llm.net.invite as invite_mod
from torrent_llm.net.identity import is_peer_id, peer_id_of
from torrent_llm.net.invite import MAX_LINK_CHARS, SCHEME, Invite, new_swarm
from torrent_llm.net.records import parse_address as real_parse_address

_STANDIN_ADDR = re.compile(r"^tcp://([a-zA-Z0-9.-]+):([0-9]{1,5})$")


def _standin_parse_address(addr: str) -> tuple[str, int]:
    m = _STANDIN_ADDR.match(addr)
    if not m:
        raise ValueError(f"invalid address: {addr}")
    host, port_str = m.groups()
    port = int(port_str)
    if not (1 <= port <= 65535):
        raise ValueError(f"port out of range: {port}")
    return host, port


@pytest.fixture(autouse=True)
def _maybe_standin_parse_address(monkeypatch: pytest.MonkeyPatch) -> None:
    try:
        real_parse_address("tcp://1.2.3.4:9")
    except NotImplementedError:
        monkeypatch.setattr(invite_mod, "parse_address", _standin_parse_address)


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _make_valid_payload() -> dict:
    return {
        "v": 1,
        "swarm": "swarm-test-12",
        "name": "Test Swarm",
        "psk": _b64url_encode(b"0123456789abcdef0123456789abcdef"),
        "pool": "a" * 64,
        "boot": ["tcp://1.2.3.4:9000"],
        "peers": [],
        "exp": 0,
    }


def _payload_to_link(payload: dict) -> str:
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return f"{SCHEME}{_b64url_encode(raw)}"


def test_round_trip():
    peer_id = peer_id_of(b"x" * 32)
    original = new_swarm(
        name="Round Trip Swarm",
        pool_hash="c" * 64,
        bootstrap=["tcp://127.0.0.1:8000", "tcp://host.example:9100"],
        peers=[peer_id],
        ttl_s=3600,
        now=2000.0,
    )
    link = original.to_link()
    assert link.startswith(SCHEME)
    decoded = Invite.from_link(link, now=2000.0)
    assert decoded == original
    assert decoded.swarm_id == original.swarm_id
    assert decoded.name == original.name
    assert decoded.psk == original.psk
    assert decoded.pool_hash == original.pool_hash
    assert decoded.bootstrap == original.bootstrap
    assert decoded.peers == original.peers
    assert decoded.expires_at == original.expires_at


def test_quotes_and_spaces_around_pasted_link():
    invite = new_swarm(
        name="Quoted Link Swarm",
        pool_hash="b" * 64,
        bootstrap=["tcp://1.1.1.1:9000"],
    )
    raw_link = invite.to_link()

    variants = [
        f"   {raw_link}   ",
        f'"{raw_link}"',
        f"'{raw_link}'",
        f'  " {raw_link} "  ',
        f"  ' {raw_link} '  ",
        f'\n\t "{raw_link}" \n',
    ]
    for variant in variants:
        parsed = Invite.from_link(variant)
        assert parsed == invite


def test_table_row_not_an_invite_link():
    with pytest.raises(ValueError, match="(?i)not an invite link"):
        Invite.from_link("https://example.com/join/abc")

    with pytest.raises(ValueError, match="(?i)not an invite link"):
        Invite.from_link("tllm://other/abc")

    with pytest.raises(ValueError, match="(?i)not an invite link"):
        Invite.from_link("")

    with pytest.raises(ValueError, match="(?i)not an invite link"):
        Invite.from_link("   ")


def test_table_row_too_long_and_exact_length():
    # Exactly MAX_LINK_CHARS gets past the length check and fails decoding as damaged
    exact_link = SCHEME + "A" * (MAX_LINK_CHARS - len(SCHEME))
    assert len(exact_link) == MAX_LINK_CHARS
    with pytest.raises(ValueError, match="(?i)damaged") as exc_info:
        Invite.from_link(exact_link)
    assert "too long" not in str(exc_info.value).lower()

    # MAX_LINK_CHARS + 1 fails length check with "too long"
    too_long_link = SCHEME + "A" * (MAX_LINK_CHARS - len(SCHEME) + 1)
    assert len(too_long_link) == MAX_LINK_CHARS + 1
    with pytest.raises(ValueError, match="(?i)too long"):
        Invite.from_link(too_long_link)


def test_table_row_damaged_base64_json_object():
    with pytest.raises(ValueError, match="(?i)damaged"):
        Invite.from_link(SCHEME + "!!not-base64!!")

    with pytest.raises(ValueError, match="(?i)damaged"):
        Invite.from_link(SCHEME + _b64url_encode(b"\xff\xfe\x00\x01"))

    not_objects = [b"42", b'"just a string"', b"[1, 2, 3]", b"true", b"null"]
    for raw in not_objects:
        with pytest.raises(ValueError, match="(?i)damaged"):
            Invite.from_link(SCHEME + _b64url_encode(raw))


def test_table_row_newer_version():
    bad_versions = [2, 0, 99, True, False, 1.0, "1", None]
    for bv in bad_versions:
        payload = _make_valid_payload()
        if bv is None:
            del payload["v"]
        else:
            payload["v"] = bv
        link = _payload_to_link(payload)
        with pytest.raises(ValueError, match="(?i)newer version"):
            Invite.from_link(link)


def test_table_row_damaged_swarm_id():
    bad_swarms = ["", "UPPERCASE", "has space", "bad@char", "a" * 65, 12345, None]
    for bad in bad_swarms:
        payload = _make_valid_payload()
        payload["swarm"] = bad
        link = _payload_to_link(payload)
        with pytest.raises(ValueError, match="(?i)damaged"):
            Invite.from_link(link)


def test_table_row_damaged_psk():
    bad_psks = [
        _b64url_encode(b"x" * 31),
        _b64url_encode(b"x" * 33),
        "!!!invalid-b64!!!",
        12345,
        None,
    ]
    for bad in bad_psks:
        payload = _make_valid_payload()
        payload["psk"] = bad
        link = _payload_to_link(payload)
        with pytest.raises(ValueError, match="(?i)damaged"):
            Invite.from_link(link)


def test_table_row_damaged_pool_hash():
    bad_pools = [
        "a" * 63,
        "a" * 65,
        "A" * 64,
        "g" * 64,
        "not-hex-at-all",
        12345,
        None,
    ]
    for bad in bad_pools:
        payload = _make_valid_payload()
        payload["pool"] = bad
        link = _payload_to_link(payload)
        with pytest.raises(ValueError, match="(?i)damaged"):
            Invite.from_link(link)


def test_table_row_no_reachable_address():
    # Missing boot
    payload = _make_valid_payload()
    del payload["boot"]
    with pytest.raises(ValueError, match="(?i)no reachable address"):
        Invite.from_link(_payload_to_link(payload))

    # Empty boot
    payload = _make_valid_payload()
    payload["boot"] = []
    with pytest.raises(ValueError, match="(?i)no reachable address"):
        Invite.from_link(_payload_to_link(payload))

    # Unparseable address
    payload = _make_valid_payload()
    payload["boot"] = ["tcp://bad host:5"]
    with pytest.raises(ValueError, match="(?i)no reachable address"):
        Invite.from_link(_payload_to_link(payload))

    # Port out of range
    payload = _make_valid_payload()
    payload["boot"] = ["tcp://1.2.3.4:0"]
    with pytest.raises(ValueError, match="(?i)no reachable address"):
        Invite.from_link(_payload_to_link(payload))

    # Non-string address
    payload = _make_valid_payload()
    payload["boot"] = [12345]
    with pytest.raises(ValueError, match="(?i)no reachable address"):
        Invite.from_link(_payload_to_link(payload))


def test_table_row_damaged_peer_id():
    valid_peer = peer_id_of(b"k" * 32)
    assert is_peer_id(valid_peer)

    bad_peers_lists = [
        ["not-a-peer-id"],
        [valid_peer, "short"],
        [valid_peer, valid_peer.upper()],
        [12345],
        [valid_peer] * 9,
    ]
    for bad_list in bad_peers_lists:
        payload = _make_valid_payload()
        payload["peers"] = bad_list
        link = _payload_to_link(payload)
        with pytest.raises(ValueError, match="(?i)damaged"):
            Invite.from_link(link)


def test_table_row_expired():
    payload = _make_valid_payload()
    payload["exp"] = 1500
    link = _payload_to_link(payload)

    # In the past -> expired
    with pytest.raises(ValueError, match="(?i)expired"):
        Invite.from_link(link, now=1500.0)

    with pytest.raises(ValueError, match="(?i)expired"):
        Invite.from_link(link, now=1500.1)

    # In the future -> ok
    parsed = Invite.from_link(link, now=1499.9)
    assert parsed.expires_at == 1500


def test_unknown_extra_keys_accepted():
    payload = _make_valid_payload()
    payload["future_key"] = "any_data"
    payload["nested_dict"] = {"enabled": True, "count": 42}
    payload["array_field"] = [1, 2, "three"]
    link = _payload_to_link(payload)
    parsed = Invite.from_link(link)
    assert parsed.swarm_id == payload["swarm"]
    assert parsed.name == payload["name"]


def test_optional_fields_defaults():
    # Name missing -> defaults to swarm id
    payload = _make_valid_payload()
    del payload["name"]
    link = _payload_to_link(payload)
    parsed = Invite.from_link(link)
    assert parsed.name == parsed.swarm_id

    # Peers missing -> defaults to empty tuple
    payload = _make_valid_payload()
    del payload["peers"]
    link = _payload_to_link(payload)
    parsed = Invite.from_link(link)
    assert parsed.peers == ()

    # Exp missing -> defaults to 0
    payload = _make_valid_payload()
    del payload["exp"]
    link = _payload_to_link(payload)
    parsed = Invite.from_link(link)
    assert parsed.expires_at == 0


def test_repr_does_not_contain_psk_secret():
    secret = b"supersecretbytes1234567890abcdef"
    invite = Invite(
        swarm_id="swarm-xyz",
        name="Private Swarm",
        psk=secret,
        pool_hash="d" * 64,
        bootstrap=("tcp://1.2.3.4:9000",),
    )
    r = repr(invite)
    assert "<hidden>" in r
    assert secret not in r.encode("latin1")
    assert _b64url_encode(secret) not in r
    assert secret.hex() not in r


def test_expiry_rules_with_injected_now():
    never_expires = Invite(
        swarm_id="swarm-never",
        name="Swarm",
        psk=b"x" * 32,
        pool_hash="0" * 64,
        bootstrap=("tcp://1.2.3.4:9000",),
        expires_at=0,
    )
    assert not never_expires.expired(now=0)
    assert not never_expires.expired(now=1000)
    assert not never_expires.expired(now=2_000_000_000)

    timed = Invite(
        swarm_id="swarm-timed",
        name="Swarm",
        psk=b"x" * 32,
        pool_hash="0" * 64,
        bootstrap=("tcp://1.2.3.4:9000",),
        expires_at=1000,
    )
    assert not timed.expired(now=500)
    assert not timed.expired(now=999.9)
    assert timed.expired(now=1000.0)
    assert timed.expired(now=1001.0)


def test_nine_bootstrap_addresses_rejected():
    nine_addrs = [f"tcp://10.0.0.{i}:9000" for i in range(1, 10)]
    assert len(nine_addrs) == 9

    # new_swarm rejects 9 addresses
    with pytest.raises(ValueError, match="(?i)no reachable address"):
        new_swarm("Test", "a" * 64, nine_addrs)

    # from_link rejects 9 addresses
    payload = _make_valid_payload()
    payload["boot"] = nine_addrs
    with pytest.raises(ValueError, match="(?i)no reachable address"):
        Invite.from_link(_payload_to_link(payload))

    # to_link rejects 9 addresses
    invite_9 = Invite(
        swarm_id="swarm-nine",
        name="Test",
        psk=b"x" * 32,
        pool_hash="a" * 64,
        bootstrap=tuple(nine_addrs),
    )
    with pytest.raises(ValueError, match="(?i)no reachable address"):
        invite_9.to_link()

    # 8 addresses are accepted
    eight_addrs = nine_addrs[:8]
    invite_8 = new_swarm("Test", "a" * 64, eight_addrs)
    assert len(invite_8.bootstrap) == 8
    link_8 = invite_8.to_link()
    assert Invite.from_link(link_8).bootstrap == tuple(eight_addrs)


def test_name_of_81_chars_rejected():
    name_81 = "x" * 81
    name_80 = "x" * 80

    # new_swarm rejects 81
    with pytest.raises(ValueError, match="(?i)damaged"):
        new_swarm(name_81, "a" * 64, ["tcp://1.2.3.4:9000"])

    # from_link rejects 81
    payload = _make_valid_payload()
    payload["name"] = name_81
    with pytest.raises(ValueError, match="(?i)damaged"):
        Invite.from_link(_payload_to_link(payload))

    # to_link rejects 81
    invite_81 = Invite(
        swarm_id="swarm-name",
        name=name_81,
        psk=b"x" * 32,
        pool_hash="a" * 64,
        bootstrap=("tcp://1.2.3.4:9000",),
    )
    with pytest.raises(ValueError, match="(?i)damaged"):
        invite_81.to_link()

    # 80 chars accepted
    invite_80 = new_swarm(name_80, "a" * 64, ["tcp://1.2.3.4:9000"])
    assert invite_80.name == name_80
    link_80 = invite_80.to_link()
    assert Invite.from_link(link_80).name == name_80


def test_exp_type_validation():
    # from_link rejects bool, float, negative, or string exp
    invalid_exps = [True, False, 1234.5, -1, "1000"]
    for bad_exp in invalid_exps:
        payload = _make_valid_payload()
        payload["exp"] = bad_exp
        link = _payload_to_link(payload)
        with pytest.raises(ValueError, match="(?i)damaged"):
            Invite.from_link(link)

    # to_link rejects bool, float, negative
    for bad_exp in [True, False, 1234.5, -1]:
        invite_bad_exp = Invite(
            swarm_id="swarm-badexp",
            name="Swarm",
            psk=b"x" * 32,
            pool_hash="a" * 64,
            bootstrap=("tcp://1.2.3.4:9000",),
            expires_at=bad_exp,
        )
        with pytest.raises(ValueError, match="(?i)damaged"):
            invite_bad_exp.to_link()


def test_new_swarm_validation():
    # Stripped name
    inv = new_swarm("  padded name  ", "a" * 64, ["tcp://1.2.3.4:9000"])
    assert inv.name == "padded name"

    # Empty name after strip
    with pytest.raises(ValueError, match="(?i)damaged"):
        new_swarm("   ", "a" * 64, ["tcp://1.2.3.4:9000"])

    # Non-string name
    with pytest.raises(ValueError, match="(?i)damaged"):
        new_swarm(123, "a" * 64, ["tcp://1.2.3.4:9000"])

    # Bad pool hash
    with pytest.raises(ValueError, match="(?i)damaged"):
        new_swarm("Valid Name", "bad-hash", ["tcp://1.2.3.4:9000"])

    # ttl_s calculation
    inv_timed = new_swarm("Valid Name", "a" * 64, ["tcp://1.2.3.4:9000"], ttl_s=120, now=500.4)
    assert inv_timed.expires_at == 620

    # ttl_s negative or bool
    with pytest.raises(ValueError, match="(?i)damaged"):
        new_swarm("Valid Name", "a" * 64, ["tcp://1.2.3.4:9000"], ttl_s=-1)
    with pytest.raises(ValueError, match="(?i)damaged"):
        new_swarm("Valid Name", "a" * 64, ["tcp://1.2.3.4:9000"], ttl_s=True)


def _valid_link() -> str:
    from torrent_llm.net.invite import new_swarm

    return new_swarm("demo", "a" * 64, ["tcp://192.168.1.20:9100"]).to_link()


@pytest.mark.parametrize("junk", ["!", "@", "*", " ", "+", "/", "="])
def test_junk_inside_the_payload_is_damage_not_silently_dropped(junk):
    from torrent_llm.net.invite import Invite

    link = _valid_link()
    cut = len("tllm://join/") + 10
    with pytest.raises(ValueError, match="(?i)damaged"):
        Invite.from_link(link[:cut] + junk + link[cut:])


def test_deeply_nested_json_is_damage_not_a_crash():
    import base64

    from torrent_llm.net.invite import Invite

    blob = base64.urlsafe_b64encode(b"[" * 2500 + b"]" * 2500).decode().rstrip("=")
    with pytest.raises(ValueError):
        Invite.from_link("tllm://join/" + blob[:3000])


def test_link_format_is_compact_and_unpadded():
    import base64
    import json

    link = _valid_link()
    payload = link[len("tllm://join/") :]
    assert "=" not in payload
    raw = base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))
    assert b" " not in raw
    assert set(json.loads(raw)) == {"v", "swarm", "name", "psk", "pool", "boot", "peers", "exp"}


def test_errors_never_contain_the_secret():
    import dataclasses

    from torrent_llm.net.invite import new_swarm

    invite = new_swarm("demo", "a" * 64, ["tcp://192.168.1.20:9100"])
    bad = dataclasses.replace(invite, psk=invite.psk + b"x")
    with pytest.raises(ValueError) as caught:
        bad.to_link()
    assert invite.psk.hex() not in str(caught.value)
    assert repr(invite.psk) not in str(caught.value)


def test_a_non_text_name_is_damage():
    import base64
    import json

    from torrent_llm.net.invite import Invite

    good = json.loads(base64.urlsafe_b64decode(_valid_link()[12:] + "=="))
    good["name"] = None
    blob = base64.urlsafe_b64encode(json.dumps(good).encode()).decode().rstrip("=")
    with pytest.raises(ValueError, match="(?i)damaged"):
        Invite.from_link("tllm://join/" + blob)
