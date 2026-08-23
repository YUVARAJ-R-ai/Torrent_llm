"""End-to-end gRPC chain tests.

Servers run in-process on OS-assigned ports, so the suite exercises real
serialisation, real sockets and the real gRPC stack without needing a second
machine. The bar is the same as for the local chain: logits must match the
monolithic model bit-exactly once they have been round-tripped through the wire.
"""

import pytest
import torch

from torrent_llm.codec import get_codec
from torrent_llm.shard import ShardRuntime, plan_even
from torrent_llm.transport import ShardClient, serve


@pytest.fixture
def chain(model_factory, num_layers):
    """A running 2-shard gRPC chain, torn down after the test."""
    specs = plan_even(num_layers, 2)
    servers, clients = [], []
    for spec in specs:
        runtime = ShardRuntime(model_factory(), spec)
        server, port = serve(runtime, get_codec("raw"), host="127.0.0.1", model_id="tiny")
        servers.append(server)
        clients.append(ShardClient(f"127.0.0.1:{port}", get_codec("raw")))

    yield clients

    for client in clients:
        client.close()
    for server in servers:
        server.stop(grace=None)


@pytest.fixture
def input_ids():
    torch.manual_seed(3)
    return torch.randint(0, 256, (1, 12))


def walk(clients, input_ids):
    """Drive the chain: token ids into the head, hidden states between hops."""
    result = clients[0].forward(input_ids, hop=0, is_token_ids=True, request_id="req-1")
    hops = [result]
    for hop, client in enumerate(clients[1:], start=1):
        result = client.forward(result.tensor, hop=hop, request_id="req-1")
        hops.append(result)
    return hops


def test_chain_over_grpc_matches_the_monolithic_model(tiny_model, chain, input_ids):
    with torch.inference_mode():
        reference = tiny_model(input_ids=input_ids).logits

    hops = walk(chain, input_ids)

    assert hops[-1].is_final
    assert torch.equal(hops[-1].tensor, reference)


def test_only_the_last_hop_reports_final(chain, input_ids):
    hops = walk(chain, input_ids)

    assert [h.is_final for h in hops] == [False, True]


def test_info_describes_what_each_node_hosts(chain, num_layers):
    head, tail = (c.info() for c in chain)

    assert (head.layer_start, head.layer_end) == (0, num_layers // 2)
    assert (tail.layer_start, tail.layer_end) == (num_layers // 2, num_layers)
    assert head.is_first and not head.is_last
    assert tail.is_last and not tail.is_first
    assert head.model_id == "tiny"


def test_hop_reports_the_bytes_it_actually_sent(chain, input_ids):
    hops = walk(chain, input_ids)

    # Hop 0 ships token ids: 12 tokens x int64.
    assert hops[0].sent_bytes == 12 * 8
    # Hop 1 ships the activation: 12 positions x hidden 64 x fp32.
    assert hops[1].sent_bytes == 12 * 64 * 4
    assert hops[1].sent_numel == 12 * 64


def test_raw_codec_reports_no_compression(chain, input_ids):
    hops = walk(chain, input_ids)

    for hop in hops:
        assert hop.compression_ratio == 1.0
        assert hop.codec == "raw"


def test_hop_separates_remote_compute_from_transport(chain, input_ids):
    hops = walk(chain, input_ids)

    for hop in hops:
        assert hop.compute_ns > 0
        assert hop.wall_ns > 0
        # Loopback, so transport is small, but the accounting must still hold.
        assert hop.transport_ns == max(0, hop.wall_ns - hop.compute_ns)


def test_explicit_position_ids_reach_every_shard(tiny_model, chain, input_ids):
    positions = torch.arange(input_ids.shape[1]).unsqueeze(0)

    result = chain[0].forward(
        input_ids, hop=0, is_token_ids=True, position_ids=positions, request_id="r"
    )
    result = chain[1].forward(result.tensor, hop=1, position_ids=positions, request_id="r")

    with torch.inference_mode():
        reference = tiny_model(input_ids=input_ids).logits
    assert torch.equal(result.tensor, reference)


def test_shifted_positions_change_the_output(chain, input_ids):
    """Guards against position_ids being silently dropped in transit."""
    default = walk(chain, input_ids)[-1].tensor

    shifted = torch.arange(5, 5 + input_ids.shape[1]).unsqueeze(0)
    r = chain[0].forward(input_ids, hop=0, is_token_ids=True, position_ids=shifted, request_id="r")
    r = chain[1].forward(r.tensor, hop=1, position_ids=shifted, request_id="r")

    assert not torch.equal(r.tensor, default)


def test_server_error_surfaces_to_the_client(chain):
    import grpc

    # Wrong width for the tail shard's layers: the failure must come back as a
    # gRPC error naming the shard, not hang or crash the server process.
    with pytest.raises(grpc.RpcError) as excinfo:
        chain[1].forward(torch.zeros(1, 4, 999), hop=1, request_id="bad")

    assert "shard1" in excinfo.value.details()


def test_server_survives_a_failed_request(tiny_model, chain, input_ids):
    import grpc

    with pytest.raises(grpc.RpcError):
        chain[1].forward(torch.zeros(1, 4, 999), hop=1, request_id="bad")

    with torch.inference_mode():
        reference = tiny_model(input_ids=input_ids).logits
    assert torch.equal(walk(chain, input_ids)[-1].tensor, reference)


# --- logits return policy ---
#
# Logits are batch x seq x vocab. On a modern tokenizer vocab dwarfs
# hidden_size (151936 vs 1024 on Qwen3-0.6B), so a full-sequence logits reply
# can be two orders of magnitude larger than the activation on the same hop --
# large enough to blow the transport cap outright at 1k context.


def test_default_returns_logits_for_every_position(tiny_model, chain, input_ids):
    with torch.inference_mode():
        reference = tiny_model(input_ids=input_ids).logits

    hops = walk(chain, input_ids)

    assert hops[-1].tensor.shape == reference.shape


def test_keep_last_returns_only_the_trailing_positions(tiny_model, chain, input_ids):
    with torch.inference_mode():
        reference = tiny_model(input_ids=input_ids).logits

    r = chain[0].forward(input_ids, hop=0, is_token_ids=True, request_id="r")
    r = chain[1].forward(r.tensor, hop=1, request_id="r", logits_keep_last=1)

    assert r.tensor.shape == (1, 1, reference.shape[-1])
    assert torch.equal(r.tensor[:, 0, :], reference[:, -1, :])


def test_keep_last_shrinks_the_reply_proportionally(chain, input_ids):
    full = chain[0].forward(input_ids, hop=0, is_token_ids=True, request_id="a")
    full = chain[1].forward(full.tensor, hop=1, request_id="a")

    trimmed = chain[0].forward(input_ids, hop=0, is_token_ids=True, request_id="b")
    trimmed = chain[1].forward(trimmed.tensor, hop=1, request_id="b", logits_keep_last=1)

    assert trimmed.received_bytes * input_ids.shape[1] == full.received_bytes


def test_keep_last_can_take_several_positions(tiny_model, chain, input_ids):
    with torch.inference_mode():
        reference = tiny_model(input_ids=input_ids).logits

    r = chain[0].forward(input_ids, hop=0, is_token_ids=True, request_id="r")
    r = chain[1].forward(r.tensor, hop=1, request_id="r", logits_keep_last=3)

    assert r.tensor.shape[1] == 3
    assert torch.equal(r.tensor, reference[:, -3:, :])


def test_keep_last_is_ignored_by_shards_that_do_not_own_the_head(chain, input_ids):
    # The head shard emits hidden states; trimming must not touch them.
    r = chain[0].forward(input_ids, hop=0, is_token_ids=True, request_id="r", logits_keep_last=1)

    assert r.tensor.shape[1] == input_ids.shape[1]


def test_oversized_reply_says_what_to_do_about_it():
    from torrent_llm.transport.server import _check_reply_size

    with pytest.raises(ValueError, match="logits_keep_last=1"):
        _check_reply_size(700 * 1024**2, is_last=True)


def test_oversized_intermediate_reply_reports_the_cap_without_the_logits_hint():
    from torrent_llm.transport.server import _check_reply_size

    with pytest.raises(ValueError, match="over the 512 MiB transport cap"):
        _check_reply_size(700 * 1024**2, is_last=False)

    _check_reply_size(1024, is_last=False)  # under the cap: no error
