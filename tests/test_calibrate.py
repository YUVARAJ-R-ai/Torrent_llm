"""Fitting a projection from data, and splicing a codec into a model to measure it."""

import pytest
import torch

from torrent_llm.codec.calibrate import CovarianceAccumulator
from torrent_llm.codec.lowrank import LowRankCodec
from torrent_llm.eval.boundary import WireTap, boundary_hook, capture_boundary, decoder_layers


def low_rank_data(rows=2000, hidden=32, true_rank=6, noise=0.01, offset=0.0, seed=0):
    gen = torch.Generator().manual_seed(seed)
    latent = torch.randn(rows, true_rank, generator=gen)
    mixing = torch.randn(true_rank, hidden, generator=gen)
    return latent @ mixing + noise * torch.randn(rows, hidden, generator=gen) + offset


def fitted(data, rank):
    acc = CovarianceAccumulator(data.shape[-1])
    acc.add(data)
    return acc.fit(rank)


def roundtrip_error(data, projection):
    codec = LowRankCodec(
        rank=projection.rank,
        down_proj=projection.down_proj,
        up_proj=projection.up_proj,
        mean=projection.mean,
    )
    x = data.reshape(1, -1, data.shape[-1])
    restored = codec.decode(codec.encode(x, request_id="t", hop=1))
    return ((restored - x).norm() / x.norm()).item()


# --- the fit ---------------------------------------------------------------------


def test_the_fit_keeps_the_directions_the_data_actually_uses():
    data = low_rank_data(true_rank=6)

    projection = fitted(data, rank=6)

    assert projection.explained_variance > 0.999
    assert roundtrip_error(data, projection) < 0.02


def test_a_random_projection_of_the_same_size_does_much_worse():
    data = low_rank_data(true_rank=6)
    gen = torch.Generator().manual_seed(1)
    q, _ = torch.linalg.qr(torch.randn(32, 6, generator=gen))
    random_codec = LowRankCodec(rank=6, down_proj=q, up_proj=q.T.contiguous())
    x = data.reshape(1, -1, 32)
    random_error = (
        (random_codec.decode(random_codec.encode(x, request_id="t", hop=1)) - x).norm() / x.norm()
    ).item()

    assert roundtrip_error(data, fitted(data, rank=6)) < random_error / 5


def test_more_directions_never_explain_less():
    data = low_rank_data(true_rank=10, noise=0.2)

    explained = [fitted(data, rank=r).explained_variance for r in (2, 4, 8, 16, 32)]

    assert explained == sorted(explained)
    assert explained[-1] == pytest.approx(1.0, abs=1e-6)


def test_keeping_every_direction_loses_nothing():
    data = low_rank_data(true_rank=32, noise=0.5)

    assert roundtrip_error(data, fitted(data, rank=32)) < 1e-4


def test_removing_a_large_constant_offset_is_what_makes_a_small_rank_work():
    # Every row carries the same big offset, as a real model's activations do.
    data = low_rank_data(true_rank=4, offset=50.0)
    centered = fitted(data, rank=4)

    codec_without_mean = LowRankCodec(
        rank=4, down_proj=centered.down_proj, up_proj=centered.up_proj
    )
    x = data.reshape(1, -1, 32)
    without = (
        (codec_without_mean.decode(codec_without_mean.encode(x, request_id="t", hop=1)) - x).norm()
        / x.norm()
    ).item()

    assert roundtrip_error(data, centered) < without / 3


def test_the_first_position_can_be_left_out_of_the_fit():
    acc = CovarianceAccumulator(8)
    batch = torch.randn(2, 5, 8)

    acc.add(batch, skip_first_position=True)

    assert acc.count == 2 * 4


def test_batches_added_separately_fit_the_same_as_added_together():
    data = low_rank_data(rows=1200)
    together = fitted(data, rank=6)
    acc = CovarianceAccumulator(32)
    for chunk in data.split(300):
        acc.add(chunk)

    apart = acc.fit(6)

    assert apart.explained_variance == pytest.approx(together.explained_variance, abs=1e-6)
    assert torch.allclose(apart.mean, together.mean, atol=1e-5)


def test_a_wrong_hidden_size_or_too_little_data_is_refused():
    acc = CovarianceAccumulator(8)
    with pytest.raises(ValueError, match="hidden size"):
        acc.add(torch.randn(4, 9))
    with pytest.raises(ValueError, match="at least two"):
        acc.fit(2)
    acc.add(torch.randn(10, 8))
    with pytest.raises(ValueError, match="rank"):
        acc.fit(9)


def test_the_variance_curve_ends_at_one_and_never_decreases():
    acc = CovarianceAccumulator(16)
    acc.add(low_rank_data(hidden=16, true_rank=5))

    curve = acc.variance_curve()

    assert curve[-1] == pytest.approx(1.0, abs=1e-6)
    assert torch.all(curve[1:] >= curve[:-1])


# --- the codec with a calibrated mean ----------------------------------------------


def test_a_calibrated_codec_survives_being_saved_and_loaded(tmp_path):
    data = low_rank_data(offset=7.0)
    projection = fitted(data, rank=6)
    codec = LowRankCodec(
        rank=6, down_proj=projection.down_proj, up_proj=projection.up_proj, mean=projection.mean
    )
    path = tmp_path / "weights.safetensors"
    codec.save_weights(path, hidden_size=32)

    reloaded = LowRankCodec(rank=6, weights_path=path)
    x = data[:20].reshape(1, 20, 32)

    assert torch.allclose(
        reloaded.decode(reloaded.encode(x, request_id="t", hop=1)),
        codec.decode(codec.encode(x, request_id="t", hop=1)),
    )
    assert reloaded.describe()["calibrated"] is True


def test_weights_saved_without_a_mean_still_load(tmp_path):
    codec = LowRankCodec(rank=4, seed=3)
    path = tmp_path / "old.safetensors"
    codec.save_weights(path, hidden_size=16)

    assert LowRankCodec(rank=4, weights_path=path)._base_mean is None


# --- splicing a codec into a model ----------------------------------------------------


def test_an_identity_hook_changes_nothing(tiny_model):
    ids = torch.randint(0, 256, (1, 10))
    with torch.inference_mode():
        plain = tiny_model(input_ids=ids).logits
        with boundary_hook(tiny_model, 3, lambda h: h):
            hooked = tiny_model(input_ids=ids).logits

    assert torch.equal(plain, hooked)


def test_the_hook_is_removed_afterwards(tiny_model):
    ids = torch.randint(0, 256, (1, 10))
    with torch.inference_mode():
        before = tiny_model(input_ids=ids).logits
        with boundary_hook(tiny_model, 3, lambda h: h * 0):
            pass
        after = tiny_model(input_ids=ids).logits

    assert torch.equal(before, after)


def test_the_boundary_must_be_inside_the_model(tiny_model):
    with pytest.raises(ValueError, match="boundary"):
        with boundary_hook(tiny_model, 0, lambda h: h):
            pass
    with pytest.raises(ValueError, match="boundary"):
        with boundary_hook(tiny_model, len(decoder_layers(tiny_model)), lambda h: h):
            pass


def test_capture_returns_the_hidden_states_that_cross_the_boundary(tiny_model):
    ids = torch.randint(0, 256, (2, 7))

    hidden = capture_boundary(tiny_model, 3, ids)

    assert hidden.shape == (2, 7, 64)


def test_a_full_rank_codec_at_the_boundary_leaves_the_logits_alone(tiny_model):
    ids = torch.randint(0, 256, (1, 12))
    acc = CovarianceAccumulator(64)
    acc.add(capture_boundary(tiny_model, 3, ids))
    projection = acc.fit(64)
    codec = LowRankCodec(
        rank=64, down_proj=projection.down_proj, up_proj=projection.up_proj, mean=projection.mean
    )
    tap = WireTap(codec)
    with torch.inference_mode():
        plain = tiny_model(input_ids=ids).logits
        with boundary_hook(tiny_model, 3, tap):
            compressed = tiny_model(input_ids=ids).logits

    assert torch.allclose(plain, compressed, atol=1e-3)
    assert tap.relative_error < 1e-3


def test_the_tap_counts_real_bytes_and_the_ratio_follows_the_rank(tiny_model):
    ids = torch.randint(0, 256, (1, 12))
    codec = LowRankCodec(rank=16, seed=0)
    tap = WireTap(codec)
    with torch.inference_mode(), boundary_hook(tiny_model, 3, tap):
        tiny_model(input_ids=ids)

    assert tap.calls == 1
    assert tap.raw_bytes == 12 * 64 * 4
    assert tap.wire_bytes == 12 * 16 * 4
    assert tap.ratio == pytest.approx(4.0)
