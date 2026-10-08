"""Simulated link: the delay math, the config block, and its effect on a real chain."""

import dataclasses

import pytest
import torch

from torrent_llm.codec import get_codec
from torrent_llm.link import LinkProfile
from torrent_llm.runner import ChainRunner, TopologyConfig
from torrent_llm.shard import ShardRuntime
from torrent_llm.transport import serve


def test_one_way_is_half_the_round_trip_plus_serialisation():
    link = LinkProfile(latency_ms=20, bandwidth_mbps=8)

    # 1 MB at 8 Mbps is one second, plus 10 ms of the 20 ms round trip.
    assert link.one_way_s(1_000_000) == pytest.approx(1.010)


def test_empty_payload_costs_only_latency():
    link = LinkProfile(latency_ms=30, bandwidth_mbps=100)

    assert link.one_way_s(0) == pytest.approx(0.015)


@pytest.mark.parametrize(
    "kwargs", [{"latency_ms": -1, "bandwidth_mbps": 10}, {"bandwidth_mbps": 0}]
)
def test_impossible_links_are_rejected(kwargs):
    with pytest.raises(ValueError):
        LinkProfile(**kwargs)


def test_link_block_is_optional():
    assert LinkProfile.from_dict(None) is None
    assert LinkProfile.from_dict({}) is None


def test_link_block_needs_a_bandwidth():
    with pytest.raises(ValueError, match="bandwidth_mbps"):
        LinkProfile.from_dict({"latency_ms": 10})


def test_link_block_rejects_unknown_keys():
    # A typo like "bandwith_mbps" would otherwise fall back to an error about a
    # missing key that hides the actual mistake.
    with pytest.raises(ValueError, match="unknown keys"):
        LinkProfile.from_dict({"bandwidth_mbps": 10, "jitter_ms": 5})


def test_topology_parses_a_link_block():
    config = TopologyConfig.from_dict(
        {
            "model_id": "tiny",
            "num_layers": 4,
            "nodes": [{"address": "127.0.0.1:1"}],
            "link": {"latency_ms": 30, "bandwidth_mbps": 100},
        }
    )

    assert config.link == LinkProfile(latency_ms=30, bandwidth_mbps=100)


@pytest.fixture
def served_chain(model_factory, num_layers):
    """Two shards on OS-assigned ports, plus the topology that describes them."""
    plan = TopologyConfig.from_dict(
        {
            "model_id": "tiny",
            "num_layers": num_layers,
            "nodes": [{"address": "127.0.0.1:0"}, {"address": "127.0.0.1:0"}],
        }
    ).shard_plan()
    servers, addresses = [], []
    for spec in plan:
        server, port = serve(
            ShardRuntime(model_factory(), spec), get_codec("raw"), host="127.0.0.1", model_id="tiny"
        )
        servers.append(server)
        addresses.append(f"127.0.0.1:{port}")

    yield TopologyConfig.from_dict(
        {
            "model_id": "tiny",
            "num_layers": num_layers,
            "nodes": [{"address": a} for a in addresses],
        }
    )

    for server in servers:
        server.stop(grace=None)


def test_shaped_hops_cost_at_least_what_the_link_charges(served_chain):
    link = LinkProfile(latency_ms=40, bandwidth_mbps=1)
    config = dataclasses.replace(served_chain, link=link)
    input_ids = torch.randint(0, 256, (1, 8))

    with ChainRunner(config) as runner:
        result = runner.forward(input_ids)

    for hop in result.hops:
        floor_s = link.one_way_s(hop.sent_bytes) + link.one_way_s(hop.received_bytes)
        assert hop.transport_ns >= floor_s * 1e9


def test_shaping_changes_the_clock_but_not_the_bytes(served_chain):
    input_ids = torch.randint(0, 256, (1, 8))
    shaped = dataclasses.replace(served_chain, link=LinkProfile(latency_ms=5, bandwidth_mbps=50))

    with ChainRunner(served_chain) as runner:
        plain = runner.forward(input_ids)
    with ChainRunner(shaped) as runner:
        slow = runner.forward(input_ids)

    assert [h.sent_bytes for h in slow.hops] == [h.sent_bytes for h in plain.hops]
    assert torch.equal(slow.logits, plain.logits)
