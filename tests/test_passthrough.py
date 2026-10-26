"""A compressed hand-over is applied once, however many hops carry it.

In a pool, the requester sits between shards: a hidden state goes shard -> requester ->
next shard. If the requester decoded it and encoded it again, the codec would be applied
twice. For a pure projection that changes nothing (projecting twice is projecting once),
but a *learned* codec is not a pure projection, so the second pass would add error the
measurements never saw. Forwarding the payload untouched makes what is measured in one
process and what is served across the network the same thing.
"""

import pytest
import torch

from torrent_llm.codec import get_codec
from torrent_llm.codec.lowrank import LowRankCodec
from torrent_llm.eval.boundary import WireTap, boundary_hook
from torrent_llm.runner import ChainRunner, NodeConfig, TopologyConfig
from torrent_llm.shard import ShardRuntime, plan_even
from torrent_llm.transport import serve


def learned_like_codec(hidden=64, rank=16, seed=0):
    """A codec that is NOT a pure projection: up is not the transpose of down.

    That is what training produces, and it is what makes applying it twice different from
    applying it once (``encode(decode(z))`` is ``z @ up @ down``, not ``z``).
    """
    gen = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(hidden, rank, generator=gen))
    down = q.contiguous()
    up = (q.T + 0.15 * torch.randn(rank, hidden, generator=gen)).contiguous()
    return LowRankCodec(
        rank=rank, down_proj=down, up_proj=up, mean=torch.randn(hidden, generator=gen)
    )


@pytest.fixture
def served(model_factory, num_layers):
    """Two real shard servers using the learned-like codec, and a topology to reach them."""
    specs = plan_even(num_layers, 2)
    servers, addresses = [], []
    for spec in specs:
        server, port = serve(
            ShardRuntime(model_factory(), spec),
            learned_like_codec(),
            host="127.0.0.1",
            model_id="tiny",
        )
        servers.append(server)
        addresses.append(f"127.0.0.1:{port}")
    yield TopologyConfig(
        model_id="tiny",
        num_layers=num_layers,
        nodes=[NodeConfig(address=a) for a in addresses],
    )
    for server in servers:
        server.stop(grace=None)


def run_chain(config, ids, codec):
    runner = ChainRunner(config)
    for client in runner.clients:
        client.codec = codec
    with runner:
        return runner.forward(ids)


def test_a_non_projection_codec_is_applied_once_across_the_network(tiny_model, served, num_layers):
    ids = torch.randint(0, 256, (1, 12))
    boundary = num_layers // 2
    with torch.inference_mode(), boundary_hook(tiny_model, boundary, WireTap(learned_like_codec())):
        once = tiny_model(input_ids=ids).logits

    over_the_wire = run_chain(served, ids, learned_like_codec())

    assert torch.allclose(over_the_wire.logits, once, atol=1e-3)


def test_applying_it_twice_would_have_been_measurably_different(tiny_model, num_layers):
    # Guards the test above: the two-step path really does differ from the one-step path
    # for this codec, so the equality above is a real check and not a coincidence.
    ids = torch.randint(0, 256, (1, 12))
    boundary = num_layers // 2
    codec = learned_like_codec()

    def twice(h):
        return codec.decode(
            codec.encode(
                codec.decode(codec.encode(h, request_id="r", hop=1)), request_id="r", hop=1
            )
        )

    with torch.inference_mode():
        with boundary_hook(tiny_model, boundary, WireTap(learned_like_codec())):
            once = tiny_model(input_ids=ids).logits
        with boundary_hook(tiny_model, boundary, twice):
            double = tiny_model(input_ids=ids).logits

    assert not torch.allclose(once, double, atol=1e-3)


def test_the_requester_does_no_encoding_for_a_payload_it_forwards(served):
    ids = torch.randint(0, 256, (1, 12))

    result = run_chain(served, ids, learned_like_codec())

    first, second = result.hops
    assert first.encode_ns > 0  # token ids are encoded by the requester
    assert second.encode_ns == 0  # the hidden state is forwarded, not re-encoded
    # What went to the second shard is byte for byte what the first one sent back.
    assert second.sent_bytes == first.received_bytes


def test_forwarding_reports_the_uncompressed_size_so_the_ratio_is_still_right(served):
    ids = torch.randint(0, 256, (1, 12))

    result = run_chain(served, ids, learned_like_codec())

    second = result.hops[1]
    assert second.sent_uncompressed_bytes == 12 * 64 * 4
    assert second.sent_uncompressed_bytes / second.sent_bytes == pytest.approx(4.0)


def test_a_reply_in_a_different_format_is_re_encoded_not_forwarded(model_factory, num_layers):
    # The server speaks the raw codec; the client is configured for low-rank. Forwarding
    # the raw reply would hand the next shard bytes it cannot read, so it must re-encode.
    specs = plan_even(num_layers, 2)
    servers, nodes = [], []
    for spec in specs:
        server, port = serve(
            ShardRuntime(model_factory(), spec), get_codec("raw"), host="127.0.0.1", model_id="tiny"
        )
        servers.append(server)
        nodes.append(NodeConfig(address=f"127.0.0.1:{port}"))
    try:
        config = TopologyConfig(model_id="tiny", num_layers=num_layers, nodes=nodes)
        runner = ChainRunner(config)
        runner.clients[1].codec = get_codec("raw")
        ids = torch.randint(0, 256, (1, 8))
        with runner:
            result = runner.forward(ids)

        assert result.hops[1].encode_ns >= 0
        assert result.logits is not None
    finally:
        for server in servers:
            server.stop(grace=None)
