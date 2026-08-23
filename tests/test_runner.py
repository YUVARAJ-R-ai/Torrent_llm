"""Chain-level behaviour: topology parsing and end-to-end runs."""

import pytest
import torch

from torrent_llm.codec import get_codec
from torrent_llm.runner import ChainRunner, TopologyConfig
from torrent_llm.shard import ShardRuntime
from torrent_llm.transport import serve


@pytest.fixture
def served_chain(model_factory, num_layers):
    """Two shards on OS-assigned ports, plus the topology that describes them."""
    base = TopologyConfig.from_dict(
        {
            "model_id": "tiny",
            "num_layers": num_layers,
            "nodes": [{"address": "127.0.0.1:0"}, {"address": "127.0.0.1:0"}],
        }
    )
    servers, addresses = [], []
    for spec in base.shard_plan():
        runtime = ShardRuntime(model_factory(), spec)
        server, port = serve(runtime, get_codec("raw"), host="127.0.0.1", model_id="tiny")
        servers.append(server)
        addresses.append(f"127.0.0.1:{port}")

    config = TopologyConfig.from_dict(
        {
            "model_id": "tiny",
            "num_layers": num_layers,
            "nodes": [{"address": a} for a in addresses],
        }
    )
    yield config

    for server in servers:
        server.stop(grace=None)


@pytest.fixture
def input_ids():
    torch.manual_seed(4)
    return torch.randint(0, 256, (1, 10))


def test_runner_reproduces_the_monolithic_logits(tiny_model, served_chain, input_ids):
    with torch.inference_mode():
        reference = tiny_model(input_ids=input_ids).logits

    with ChainRunner(served_chain) as runner:
        result = runner.forward(input_ids)

    assert torch.equal(result.logits, reference)


def test_runner_records_one_hop_per_node(served_chain, input_ids):
    with ChainRunner(served_chain) as runner:
        result = runner.forward(input_ids)

    assert len(result.hops) == 2
    assert [h.hop for h in result.hops] == [0, 1]
    assert result.total_sent_bytes == sum(h.sent_bytes for h in result.hops)


def test_all_hops_of_a_request_share_one_id(served_chain, input_ids):
    with ChainRunner(served_chain) as runner:
        result = runner.forward(input_ids)

    assert len({result.request_id}) == 1
    assert result.total_wall_ns >= result.total_compute_ns > 0


def test_describe_reports_the_whole_chain(served_chain, num_layers):
    with ChainRunner(served_chain) as runner:
        rows = runner.describe()

    assert [r["layers"] for r in rows] == [
        f"[0:{num_layers // 2})",
        f"[{num_layers // 2}:{num_layers})",
    ]


def test_greedy_decode_extends_the_sequence(served_chain, input_ids):
    with ChainRunner(served_chain) as runner:
        ids, passes = runner.generate(input_ids, max_new_tokens=3)

    assert ids.shape == (1, input_ids.shape[1] + 3)
    assert len(passes) == 3


def test_greedy_decode_matches_the_reference_model(tiny_model, served_chain, input_ids):
    with ChainRunner(served_chain) as runner:
        ids, _ = runner.generate(input_ids, max_new_tokens=3)

    expected = input_ids
    with torch.inference_mode():
        for _ in range(3):
            nxt = tiny_model(input_ids=expected).logits[:, -1, :].argmax(-1, keepdim=True)
            expected = torch.cat([expected, nxt], dim=1)

    assert torch.equal(ids, expected)


# --- topology config ---


def test_topology_requires_its_core_keys():
    with pytest.raises(ValueError, match=r"missing required keys: \['model_id', 'nodes'\]"):
        TopologyConfig.from_dict({"num_layers": 4})


def test_topology_splits_evenly_by_default():
    config = TopologyConfig.from_dict(
        {
            "model_id": "m",
            "num_layers": 32,
            "nodes": [{"address": "a:1"}, {"address": "b:1"}],
        }
    )

    assert [(s.start, s.end) for s in config.shard_plan()] == [(0, 16), (16, 32)]


def test_explicit_boundaries_override_the_even_split():
    config = TopologyConfig.from_dict(
        {
            "model_id": "m",
            "num_layers": 32,
            "boundaries": [20],
            "nodes": [{"address": "a:1"}, {"address": "b:1"}],
        }
    )

    assert [(s.start, s.end) for s in config.shard_plan()] == [(0, 20), (20, 32)]


def test_boundaries_that_disagree_with_the_node_count_are_rejected():
    config = TopologyConfig.from_dict(
        {
            "model_id": "m",
            "num_layers": 32,
            "boundaries": [10, 20],
            "nodes": [{"address": "a:1"}, {"address": "b:1"}],
        }
    )

    with pytest.raises(ValueError, match="2 nodes but the plan produced 3 shards"):
        config.shard_plan()


def test_codec_can_be_a_name_or_a_block_with_args():
    plain = TopologyConfig.from_dict(
        {"model_id": "m", "num_layers": 4, "nodes": [{"address": "a:1"}], "codec": "raw"}
    )
    with_args = TopologyConfig.from_dict(
        {
            "model_id": "m",
            "num_layers": 4,
            "nodes": [{"address": "a:1"}],
            "codec": {"name": "raw", "wire_dtype": "float16"},
        }
    )

    assert plain.codec == "raw" and plain.codec_args == {}
    assert with_args.codec == "raw" and with_args.codec_args == {"wire_dtype": "float16"}


def test_node_address_splits_into_host_and_port():
    config = TopologyConfig.from_dict(
        {"model_id": "m", "num_layers": 4, "nodes": [{"address": "192.168.1.10:50051"}]}
    )

    assert config.nodes[0].host == "192.168.1.10"
    assert config.nodes[0].port == 50051


@pytest.mark.parametrize("name", ["local-2shard.yaml", "lan-2machine.yaml"])
def test_shipped_configs_parse_and_plan(name):
    """The configs in the repo must actually work — a typo here wastes a rig session."""
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "configs" / name
    config = TopologyConfig.from_file(path)

    specs = config.shard_plan()
    assert len(specs) == len(config.nodes)
    assert specs[0].is_first and specs[-1].is_last
    assert getattr(torch, config.dtype) is not None
