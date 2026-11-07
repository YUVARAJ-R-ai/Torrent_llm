"""Unit tests for Int8Codec.

Tests the contract specified in Int8Codec docstring:
- decode(encode(x)) preserves shape and dtype across float32, float16, bfloat16 and ranks >= 1
- relative error < 0.01 for standard normal activations
- outlier channel preservation (< 1e-3 relative error) without degrading other channels (< 0.01)
- all-zero rows decode to zeros
- outliers=0 and outliers >= H edge cases
- integer tensor bit-exact passthrough
- payload byte layout formula verification (N*H + 2N + 2Nk + 4k)
- header metadata preservation (request_id, hop)
- empty sequence (1, 0, 16) handling
- single-row activations
- codec registry and describe() inspection
- strict decode validation against malformed or malicious network payloads
- tiny rows loss of precision and huge rows float16 saturation
- multi-row outlier channels in different columns per row
"""

import pytest
import torch

from torrent_llm.codec import available_codecs, get_codec
from torrent_llm.codec.int8 import Int8Codec
from torrent_llm.codec.raw import tensor_to_bytes
from torrent_llm.wire import ActivationHeader, ActivationMessage


def _rel_err(x: torch.Tensor, y: torch.Tensor) -> float:
    return (torch.norm((x - y).float()) / torch.norm(x.float())).item()


def test_int8_codec_registered():
    assert "int8" in available_codecs()
    codec = get_codec("int8", outliers=4)
    assert isinstance(codec, Int8Codec)
    assert codec.outliers == 4


def test_int8_describe():
    codec = Int8Codec(outliers=6)
    assert codec.describe() == {"codec": "int8", "outliers": 6}


def test_int8_rejects_negative_outliers():
    with pytest.raises(ValueError, match="outliers must be >= 0"):
        Int8Codec(outliers=-1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(64,), (4, 64), (2, 7, 64), (2, 3, 4, 64)])
def test_int8_preserves_shape_and_dtype(dtype, shape):
    codec = Int8Codec(outliers=4)
    x = torch.randn(shape, dtype=dtype)
    msg = codec.encode(x, request_id="req-1", hop=0)
    restored = codec.decode(msg)

    assert restored.shape == x.shape
    assert restored.dtype == x.dtype


def test_int8_relative_error_on_gaussian():
    torch.manual_seed(42)
    x = torch.randn(4, 37, 1024, dtype=torch.float32)
    codec = Int8Codec(outliers=8)
    msg = codec.encode(x, request_id="req-gauss", hop=1)
    y = codec.decode(msg)

    err = _rel_err(x, y)
    assert err < 0.01


def test_int8_outlier_channel_preservation():
    torch.manual_seed(42)
    x = torch.randn(4, 37, 1024, dtype=torch.float32)
    outlier_col = 42
    x[:, :, outlier_col] *= 1000.0

    codec = Int8Codec(outliers=8)
    msg = codec.encode(x, request_id="req-outlier", hop=2)
    y = codec.decode(msg)

    # The outlier channel survives with relative error < 1e-3
    outlier_orig = x[:, :, outlier_col]
    outlier_recon = y[:, :, outlier_col]
    outlier_err = _rel_err(outlier_orig, outlier_recon)
    assert outlier_err < 1e-3

    # Overall relative error < 0.01
    overall_err = _rel_err(x, y)
    assert overall_err < 0.01

    # Non-outlier channels are not ruined (rel error < 0.01)
    mask = torch.ones(1024, dtype=torch.bool)
    mask[outlier_col] = False
    non_outlier_err = _rel_err(x[:, :, mask], y[:, :, mask])
    assert non_outlier_err < 0.01


def test_int8_multi_row_different_outlier_columns():
    """Different rows having peak activations in different columns are both captured."""
    torch.manual_seed(123)
    x = torch.randn(3, 16, dtype=torch.float32)
    x[0, 2] = 500.0
    x[1, 5] = 600.0
    x[2, 9] = 700.0

    codec = Int8Codec(outliers=3)
    msg = codec.encode(x, request_id="req-diff-cols", hop=0)
    y = codec.decode(msg)

    # Columns 2, 5, 9 should be captured as the top-3 outliers
    assert msg.header.codec_meta["k"] == 3
    # Check that each outlier spike was recovered with float16 precision
    assert abs(y[0, 2].item() - 500.0) / 500.0 < 1e-3
    assert abs(y[1, 5].item() - 600.0) / 600.0 < 1e-3
    assert abs(y[2, 9].item() - 700.0) / 700.0 < 1e-3

    # Overall error remains low
    assert _rel_err(x, y) < 0.01


def test_int8_all_zero_rows_decode_to_zeros():
    codec = Int8Codec(outliers=4)
    x = torch.zeros(3, 128, dtype=torch.float32)
    msg = codec.encode(x, request_id="req-zeros", hop=0)
    restored = codec.decode(msg)

    assert torch.equal(restored, x)


def test_int8_outliers_zero():
    codec = Int8Codec(outliers=0)
    x = torch.randn(2, 64, dtype=torch.float32)
    msg = codec.encode(x, request_id="req-k0", hop=1)

    assert msg.header.codec_meta["k"] == 0
    restored = codec.decode(msg)
    assert _rel_err(x, restored) < 0.02


def test_int8_outliers_greater_than_or_equal_to_cols():
    # When outliers >= H, all channels travel in float16
    codec = Int8Codec(outliers=100)
    x = torch.randn(2, 32, dtype=torch.float32)
    msg = codec.encode(x, request_id="req-kgeh", hop=1)

    assert msg.header.codec_meta["k"] == 32
    restored = codec.decode(msg)
    # Since all channels are float16, relative error is within float16 roundoff (< 1e-3)
    assert _rel_err(x, restored) < 1e-3


@pytest.mark.parametrize("dtype", [torch.int64, torch.int32, torch.uint8])
def test_int8_integer_tensors_round_trip_bit_exactly(dtype):
    codec = Int8Codec(outliers=4)
    high = 256 if dtype == torch.uint8 else 150000
    ids = torch.randint(0, high, (2, 16), dtype=dtype)
    msg = codec.encode(ids, request_id="req-int", hop=0)

    assert msg.header.codec_meta.get("passthrough") is True
    restored = codec.decode(msg)
    assert torch.equal(restored, ids)
    assert restored.dtype == dtype


def test_int8_payload_bytes_matches_formula():
    n_rows, h_cols, k_outliers = 5, 64, 6
    codec = Int8Codec(outliers=k_outliers)
    x = torch.randn(n_rows, h_cols, dtype=torch.float32)
    msg = codec.encode(x, request_id="req-size", hop=1)

    expected = n_rows * h_cols + 4 * n_rows + 4 * n_rows * k_outliers + 4 * k_outliers
    assert msg.payload_bytes == expected


def test_int8_header_preserves_request_id_and_hop():
    codec = Int8Codec(outliers=2)
    x = torch.randn(1, 16, dtype=torch.float32)
    msg = codec.encode(x, request_id="req-unique-99", hop=7)

    assert msg.header.request_id == "req-unique-99"
    assert msg.header.hop == 7
    assert msg.header.codec == "int8"


def test_int8_empty_sequence_round_trips():
    codec = Int8Codec(outliers=4)
    x = torch.zeros(1, 0, 16, dtype=torch.float32)
    msg = codec.encode(x, request_id="req-empty", hop=0)

    # For N=0, H=16, k=min(4, 16)=4: payload is int32[4] = 16 bytes
    assert msg.payload_bytes == 4 * 4
    restored = codec.decode(msg)
    assert restored.shape == (1, 0, 16)
    assert restored.dtype == torch.float32


def test_int8_single_row():
    codec = Int8Codec(outliers=2)
    # 1D single row tensor
    x1 = torch.randn(32, dtype=torch.float32)
    msg1 = codec.encode(x1, request_id="req-1d", hop=0)
    assert codec.decode(msg1).shape == (32,)

    # 2D single row tensor
    x2 = torch.randn(1, 32, dtype=torch.float32)
    msg2 = codec.encode(x2, request_id="req-2d", hop=0)
    assert codec.decode(msg2).shape == (1, 32)


def test_int8_rows_of_any_magnitude_keep_their_precision():
    """float32 scales: tiny and huge rows are as accurate as ordinary ones."""
    codec = Int8Codec(outliers=2)
    for magnitude in (1e-6, 1e-3, 1.0, 1e4, 1e8):
        x = torch.randn(3, 7, 256, generator=torch.Generator().manual_seed(1)) * magnitude
        y = codec.decode(codec.encode(x, request_id="r", hop=0))
        rel = (x - y).norm() / x.norm()
        assert rel < 0.01, (magnitude, float(rel))


def test_int8_refuses_non_finite_hidden_states():
    codec = Int8Codec()
    for bad in (float("nan"), float("inf"), float("-inf")):
        x = torch.randn(2, 16)
        x[1, 3] = bad
        with pytest.raises(ValueError, match="NaN or infinite"):
            codec.encode(x, request_id="r", hop=0)


def test_int8_decode_rejects_invalid_indices():
    rows, cols, k = 2, 8, 3
    base_indices = torch.tensor([1, 3, 5], dtype=torch.int32)
    outliers_vals = torch.zeros((rows, k), dtype=torch.float32)
    scales = torch.ones((rows,), dtype=torch.float32)
    q = torch.zeros((rows, cols), dtype=torch.int8)

    def make_msg(indices_tensor: torch.Tensor) -> ActivationMessage:
        payload = (
            tensor_to_bytes(indices_tensor)
            + tensor_to_bytes(outliers_vals)
            + tensor_to_bytes(scales)
            + tensor_to_bytes(q)
        )
        header = ActivationHeader(
            request_id="r",
            hop=0,
            codec="int8",
            dtype="float32",
            shape=(rows, cols),
            codec_meta={"v": 1, "rows": rows, "cols": cols, "k": k},
        )
        return ActivationMessage(header=header, payload=payload)

    codec = Int8Codec()

    # Valid indices pass
    codec.decode(make_msg(base_indices))

    # Negative index
    with pytest.raises(ValueError, match="indices out of range"):
        codec.decode(make_msg(torch.tensor([-1, 3, 5], dtype=torch.int32)))

    # Index >= cols
    with pytest.raises(ValueError, match="indices out of range"):
        codec.decode(make_msg(torch.tensor([1, 3, 8], dtype=torch.int32)))

    # Unsorted indices
    with pytest.raises(ValueError, match="indices must be strictly increasing"):
        codec.decode(make_msg(torch.tensor([1, 5, 3], dtype=torch.int32)))

    # Duplicated indices
    with pytest.raises(ValueError, match="indices must be strictly increasing"):
        codec.decode(make_msg(torch.tensor([1, 3, 3], dtype=torch.int32)))


def test_int8_decode_rejects_bool_or_float_version():
    """Explicitly verify that v=True (bool) and v=1.0 (float) are rejected."""
    codec = Int8Codec(outliers=1)
    x = torch.randn(2, 4, dtype=torch.float32)
    msg = codec.encode(x, request_id="r", hop=0)

    # Bool version v=True
    meta_bool = dict(msg.header.codec_meta)
    meta_bool["v"] = True
    msg_bool = ActivationMessage(
        header=ActivationHeader(
            request_id="r",
            hop=0,
            codec="int8",
            dtype=msg.header.dtype,
            shape=msg.header.shape,
            codec_meta=meta_bool,
        ),
        payload=msg.payload,
    )
    with pytest.raises(ValueError, match="unsupported or invalid codec version"):
        codec.decode(msg_bool)

    # Float version v=1.0
    meta_float = dict(msg.header.codec_meta)
    meta_float["v"] = 1.0
    msg_float = ActivationMessage(
        header=ActivationHeader(
            request_id="r",
            hop=0,
            codec="int8",
            dtype=msg.header.dtype,
            shape=msg.header.shape,
            codec_meta=meta_float,
        ),
        payload=msg.payload,
    )
    with pytest.raises(ValueError, match="unsupported or invalid codec version"):
        codec.decode(msg_float)


def test_int8_extreme_outlier_survives_exactly():
    """An outlier far beyond float16's range travels intact (float32 outliers)."""
    codec = Int8Codec(outliers=1)
    x = torch.randn(2, 8, dtype=torch.float32)
    x[0, 3] = 1e6
    restored = codec.decode(codec.encode(x, request_id="r", hop=0))
    assert restored[0, 3] == 1e6
    assert torch.isfinite(restored).all()


def test_int8_decode_refuses_non_finite_scales():
    codec = Int8Codec(outliers=0)
    msg = codec.encode(torch.randn(2, 8), request_id="r", hop=0)
    payload = bytearray(msg.payload)
    payload[0:4] = torch.tensor([float("nan")], dtype=torch.float32).numpy().tobytes()
    with pytest.raises(ValueError, match="NaN or infinite"):
        codec.decode(ActivationMessage(header=msg.header, payload=bytes(payload)))


def test_int8_decoder_produces_writable_tensor():
    """Decoded tensor is independent of the message buffer and writable."""
    codec = Int8Codec(outliers=2)
    x = torch.randn(2, 8, dtype=torch.float32)
    msg = codec.encode(x, request_id="r", hop=0)
    decoded = codec.decode(msg)

    # Must be writable without RuntimeError
    decoded[0, 0] = 999.0
    assert decoded[0, 0] == 999.0


def _encoded(rows: int = 3, cols: int = 16, k: int = 2) -> ActivationMessage:
    return Int8Codec(outliers=k).encode(torch.randn(rows, cols), request_id="r", hop=0)


def _with_meta(msg: ActivationMessage, **changes) -> ActivationMessage:
    import dataclasses

    meta = {**msg.header.codec_meta, **changes}
    return ActivationMessage(
        header=dataclasses.replace(msg.header, codec_meta=meta), payload=msg.payload
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"k": 3},  # payload no longer matches the length implied by k
        {"k": -1},
        {"k": 17},  # more outliers than columns
        {"rows": 4},  # rows * cols != numel
        {"cols": 8},  # cols != shape[-1]
        {"rows": -3, "cols": -16},
        {"rows": "3"},
        {"k": 2.0},
        {"k": True},
        {"v": 2},
        {"v": None},
    ],
)
def test_int8_decode_rejects_tampered_metadata(changes):
    with pytest.raises(ValueError):
        Int8Codec().decode(_with_meta(_encoded(), **changes))


@pytest.mark.parametrize("delta", [-1, 1, -100])
def test_int8_decode_rejects_wrong_payload_length(delta):
    msg = _encoded()
    payload = msg.payload[:delta] if delta < 0 else msg.payload + b"\0" * delta
    with pytest.raises(ValueError, match="payload length"):
        Int8Codec().decode(ActivationMessage(header=msg.header, payload=payload))


def test_int8_passthrough_rejects_a_payload_of_the_wrong_size():
    msg = Int8Codec().encode(torch.arange(6).reshape(2, 3), request_id="r", hop=0)
    assert msg.header.codec_meta == {"v": 1, "passthrough": True}
    with pytest.raises(ValueError):
        Int8Codec().decode(ActivationMessage(header=msg.header, payload=msg.payload[:-1]))


def test_int8_is_about_four_times_smaller_than_float32():
    x = torch.randn(1, 128, 1024)
    msg = Int8Codec().encode(x, request_id="r", hop=0)
    assert 3.8 < (x.numel() * 4) / msg.payload_bytes < 4.0
