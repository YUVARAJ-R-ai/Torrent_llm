"""Round-trip contract every codec must satisfy.

Value fidelity is codec-specific and belongs to the quality harness (issue #10);
what is tested here is the structural contract transport relies on — shape and
dtype survive the trip, and the payload length is what the profiler will see.
"""

import pytest
import torch

from torrent_llm.codec import available_codecs, get_codec
from torrent_llm.wire import ActivationHeader, ActivationMessage


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_raw_round_trip_is_exact(dtype):
    codec = get_codec("raw")
    original = torch.randn(2, 7, 64).to(dtype)

    msg = codec.encode(original, request_id="req-1", hop=0)
    restored = codec.decode(msg)

    assert restored.shape == original.shape
    assert restored.dtype == original.dtype
    # "raw" is the control condition: it must be bit-exact, not merely close.
    assert torch.equal(restored, original)


def test_raw_payload_is_the_uncompressed_size():
    codec = get_codec("raw")
    tensor = torch.randn(1, 128, 512, dtype=torch.float16)

    msg = codec.encode(tensor, request_id="req-1", hop=0)

    assert msg.payload_bytes == tensor.numel() * tensor.element_size()
    assert msg.header.logical_numel == tensor.numel()


def test_raw_can_cast_to_a_narrower_wire_dtype():
    codec = get_codec("raw", wire_dtype="float16")
    tensor = torch.randn(4, 32, dtype=torch.float32)

    msg = codec.encode(tensor, request_id="req-1", hop=2)
    restored = codec.decode(msg)

    assert msg.header.dtype == "float16"
    assert restored.dtype == torch.float16
    assert msg.payload_bytes == tensor.numel() * 2
    torch.testing.assert_close(restored.float(), tensor, rtol=1e-2, atol=1e-2)


def test_header_carries_request_identity_across_the_hop():
    codec = get_codec("raw")

    msg = codec.encode(torch.zeros(1, 4, 8), request_id="req-abc", hop=3)

    assert msg.header.request_id == "req-abc"
    assert msg.header.hop == 3
    assert msg.header.codec == "raw"


def test_unknown_codec_names_the_registered_ones():
    with pytest.raises(KeyError, match="raw"):
        get_codec("does-not-exist")

    assert "raw" in available_codecs()


def test_header_rejects_dtypes_transport_cannot_frame():
    with pytest.raises(ValueError, match="unsupported wire dtype"):
        ActivationHeader(request_id="r", hop=0, codec="raw", dtype="complex64", shape=(1,))


def test_empty_payload_round_trips_as_an_empty_tensor():
    # A zero-length sequence is a legitimate edge case for a batching runner.
    codec = get_codec("raw")
    msg = codec.encode(torch.zeros(1, 0, 16), request_id="r", hop=0)

    assert msg.payload_bytes == 0
    assert codec.decode(msg).shape == (1, 0, 16)


def test_message_payload_size_is_independent_of_logical_size():
    # The property a compressor exploits: the wire payload may be far smaller
    # than the logical tensor. Transport must never assume they match.
    header = ActivationHeader(
        request_id="r", hop=0, codec="pretend", dtype="float16", shape=(1, 256, 1024)
    )
    msg = ActivationMessage(header=header, payload=b"\x00" * 512)

    assert msg.header.logical_numel == 262144
    assert msg.payload_bytes == 512
