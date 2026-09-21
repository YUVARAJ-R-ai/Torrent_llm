"""Unit tests for LowRankCodec (Issue #8).

Tests the structural and compression contracts for low-rank activation compression:
- Codec registry integration
- Proportional payload compression on the wire
- Round-trip preservation of shape and dtype (float32, float16, bfloat16)
- Orthonormal projection stability (energy boundedness and idempotency)
- Multi-process deterministic synchronization via shared seed
- Hop 0 token ID passthrough
- Empty sequence handling
- Explicit weights and checkpoint save/load
- Wire dtype downcasting
"""

from pathlib import Path

import pytest
import torch

from torrent_llm.codec import available_codecs, get_codec
from torrent_llm.codec.lowrank import LowRankCodec


def test_lowrank_codec_registered():
    assert "lowrank" in available_codecs()
    codec = get_codec("lowrank", rank=64)
    assert isinstance(codec, LowRankCodec)
    assert codec.rank == 64


def test_lowrank_compresses_wire_payload_proportionally():
    raw_codec = get_codec("raw")
    rank = 64
    hidden = 512
    lowrank_codec = get_codec("lowrank", rank=rank)

    tensor = torch.randn(2, 32, hidden, dtype=torch.float32)

    raw_msg = raw_codec.encode(tensor, request_id="req-1", hop=1)
    lr_msg = lowrank_codec.encode(tensor, request_id="req-1", hop=1)

    expected_ratio = hidden / rank  # 512 / 64 = 8.0
    actual_ratio = raw_msg.payload_bytes / lr_msg.payload_bytes
    assert actual_ratio == pytest.approx(expected_ratio, rel=1e-3)
    assert lr_msg.header.codec_meta["rank"] == rank
    assert lr_msg.header.logical_numel == tensor.numel()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_lowrank_preserves_shape_and_dtype(dtype):
    codec = get_codec("lowrank", rank=32)
    tensor = torch.randn(2, 8, 128, dtype=dtype)

    msg = codec.encode(tensor, request_id="req-test", hop=1)
    restored = codec.decode(msg)

    assert restored.shape == tensor.shape
    assert restored.dtype == tensor.dtype


def test_lowrank_orthonormal_projection_properties():
    rank = 64
    hidden = 256
    codec = get_codec("lowrank", rank=rank, seed=99)
    tensor = torch.randn(4, 16, hidden, dtype=torch.float32)

    msg = codec.encode(tensor, request_id="req-p", hop=1)
    p_x = codec.decode(msg)

    # 1. Energy bound: ||P(x)||_2 <= ||x||_2 for orthogonal projection
    norm_orig = torch.linalg.vector_norm(tensor, dim=-1)
    norm_proj = torch.linalg.vector_norm(p_x, dim=-1)
    assert torch.all(norm_proj <= norm_orig + 1e-4)

    # 2. Idempotency: P(P(x)) == P(x)
    msg2 = codec.encode(p_x, request_id="req-p2", hop=2)
    p_p_x = codec.decode(msg2)
    torch.testing.assert_close(p_p_x, p_x, rtol=1e-3, atol=1e-3)


def test_lowrank_deterministic_seed_across_processes():
    # Two independently constructed codecs with the same seed must encode and decode identically
    codec_sender = LowRankCodec(rank=64, seed=42)
    codec_receiver = LowRankCodec(rank=64, seed=42)

    tensor = torch.randn(1, 10, 128, dtype=torch.float32)

    msg = codec_sender.encode(tensor, request_id="req-seed", hop=1)
    restored = codec_receiver.decode(msg)

    assert restored.shape == tensor.shape
    # Decoding locally with sender must be bit-identical to decoding with receiver
    sender_restored = codec_sender.decode(msg)
    assert torch.equal(restored, sender_restored)


def test_lowrank_hop0_token_ids_passthrough():
    codec = get_codec("lowrank", rank=64)
    token_ids = torch.randint(0, 32000, (2, 24), dtype=torch.long)

    msg = codec.encode(token_ids, request_id="req-hop0", hop=0)
    assert msg.header.codec_meta.get("passthrough") is True
    assert msg.payload_bytes == token_ids.numel() * token_ids.element_size()

    restored = codec.decode(msg)
    assert restored.shape == token_ids.shape
    assert restored.dtype == torch.long
    assert torch.equal(restored, token_ids)


def test_lowrank_empty_sequence():
    codec = get_codec("lowrank", rank=32)
    empty_tensor = torch.zeros(1, 0, 128, dtype=torch.float32)

    msg = codec.encode(empty_tensor, request_id="req-empty", hop=1)
    assert msg.payload_bytes == 0

    restored = codec.decode(msg)
    assert restored.shape == (1, 0, 128)
    assert restored.dtype == torch.float32


def test_lowrank_wire_dtype_casting():
    codec = get_codec("lowrank", rank=64, wire_dtype="float16")
    tensor = torch.randn(1, 16, 256, dtype=torch.float32)

    msg = codec.encode(tensor, request_id="req-cast", hop=1)
    # Wire payload uses 2 bytes per element (fp16) instead of 4 (fp32)
    expected_bytes = 1 * 16 * 64 * 2
    assert msg.payload_bytes == expected_bytes
    assert msg.header.dtype == "float32"

    restored = codec.decode(msg)
    assert restored.dtype == torch.float32
    assert restored.shape == tensor.shape


def test_lowrank_save_and_load_weights(tmp_path: Path):
    hidden = 128
    rank = 32
    codec1 = LowRankCodec(rank=rank, seed=777)

    save_file = tmp_path / "projection_weights.pt"
    codec1.save_weights(save_file, hidden_size=hidden)
    assert save_file.exists()

    codec2 = LowRankCodec(rank=rank, weights_path=save_file)
    tensor = torch.randn(2, 4, hidden, dtype=torch.float32)

    msg = codec1.encode(tensor, request_id="req-save", hop=1)
    restored = codec2.decode(msg)

    torch.testing.assert_close(restored, codec1.decode(msg))


def test_lowrank_describe():
    codec = get_codec("lowrank", rank=128, seed=42)
    desc = codec.describe()
    assert desc["codec"] == "lowrank"
    assert desc["rank"] == 128
    assert desc["seed"] == 42
    assert desc["wire_dtype"] is None
