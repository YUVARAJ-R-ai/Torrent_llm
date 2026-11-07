"""Codec profiles: messages name their profile, receivers decode with their own copy."""

from __future__ import annotations

import pytest
import torch

from torrent_llm.codec.profiles import DEFAULT_PROFILES, CodecProfile, CodecSet, UnknownProfile


def test_default_profiles_round_trip_and_label_the_wire_with_the_profile_id():
    codecs = CodecSet.from_config(None)
    assert set(codecs.ids) == set(DEFAULT_PROFILES)
    x = torch.randn(1, 5, 64)
    msg = codecs.codec("bf16").encode(x, request_id="r", hop=1)
    assert msg.header.codec == "bf16"
    assert msg.payload_bytes == x.numel() * 2
    y = codecs.decode(msg)
    assert y.shape == x.shape
    assert torch.allclose(y.float(), x, rtol=1e-2, atol=1e-2)
    exact = codecs.decode(codecs.codec("raw32").encode(x, request_id="r", hop=1))
    assert torch.equal(exact, x)


def test_an_unknown_profile_is_refused_not_guessed():
    codecs = CodecSet.from_config({"bf16": {"name": "raw", "wire_dtype": "bfloat16"}})
    other = CodecSet.from_config(None)
    msg = other.codec("raw32").encode(torch.ones(2, 3), request_id="r", hop=0)
    with pytest.raises(UnknownProfile):
        codecs.decode(msg)


def test_signature_depends_on_settings_but_not_on_where_weights_live():
    a = CodecSet([CodecProfile("p", "lowrank", {"rank": 8, "weights_path": "/x/w.safetensors"})])
    b = CodecSet(
        [CodecProfile("p", "lowrank", {"rank": 8, "weights_path": "C:\\y\\w.safetensors"})]
    )
    c = CodecSet([CodecProfile("p", "lowrank", {"rank": 16, "weights_path": "/x/w.safetensors"})])
    assert a.signature() == b.signature() != c.signature()


def test_bad_profile_definitions_are_rejected():
    with pytest.raises(ValueError):
        CodecProfile("Bad Id", "raw")
    with pytest.raises(ValueError):
        CodecSet([])
    with pytest.raises(ValueError):
        CodecSet([CodecProfile("a", "raw"), CodecProfile("a", "raw")])
    with pytest.raises(ValueError):
        CodecSet.from_config({"a": {"wire_dtype": "bfloat16"}})


def test_token_ids_cross_every_profile_exactly():
    codecs = CodecSet.from_config(None)
    ids = torch.tensor([[1, 151_000, 7]])
    for pid in codecs.ids:
        out = codecs.decode(codecs.codec(pid).encode(ids, request_id="r", hop=0))
        assert torch.equal(out, ids), pid
