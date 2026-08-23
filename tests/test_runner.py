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


def test_weights_override_the_even_split():
    config = TopologyConfig.from_dict(
        {
            "model_id": "m",
            "num_layers": 30,
            "weights": [1.0, 2.0],
            "nodes": [{"address": "a:1"}, {"address": "b:1"}],
        }
    )

    assert [(s.start, s.end) for s in config.shard_plan()] == [(0, 10), (10, 30)]


def test_weights_that_disagree_with_the_node_count_are_rejected():
    config = TopologyConfig.from_dict(
        {
            "model_id": "m",
            "num_layers": 32,
            "weights": [1.0, 1.0, 1.0],
            "nodes": [{"address": "a:1"}, {"address": "b:1"}],
        }
    )

    with pytest.raises(ValueError, match="2 nodes but the plan produced 3 shards"):
        config.shard_plan()


def test_boundaries_and_weights_together_is_rejected_as_ambiguous():
    with pytest.raises(ValueError, match="both 'boundaries' and 'weights'"):
        TopologyConfig.from_dict(
            {
                "model_id": "m",
                "num_layers": 32,
                "boundaries": [16],
                "weights": [1.0, 1.0],
                "nodes": [{"address": "a:1"}, {"address": "b:1"}],
            }
        )


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


def test_runner_profiles_every_hop(served_chain, input_ids, tmp_path):
    """The profiler must see one record per hop, with the shape actually sent."""
    from torrent_llm.profile import HopProfiler, RunMetadata

    metadata = RunMetadata(run_id="t", model_id="tiny", dtype="float32", codec="raw", num_shards=2)
    with HopProfiler(path=tmp_path / "p.jsonl", metadata=metadata) as profiler:
        with ChainRunner(served_chain, profiler=profiler) as runner:
            runner.forward(input_ids)
        records = profiler.records

    assert [r.hop for r in records] == [0, 1]
    # Hop 0 sends token ids, so there is no hidden dimension yet.
    assert records[0].hidden_size == 0
    assert records[0].seq_len == input_ids.shape[1]
    # Hop 1 sends the activation.
    assert records[1].hidden_size == 64
    assert records[1].sent_bytes == input_ids.shape[1] * 64 * 4


def test_generate_labels_the_first_pass_prefill_and_the_rest_decode(served_chain, input_ids):
    from torrent_llm.profile import HopProfiler

    with HopProfiler() as profiler:
        with ChainRunner(served_chain, profiler=profiler) as runner:
            runner.generate(input_ids, max_new_tokens=3)
        phases = [r.phase for r in profiler.records]

    # 3 passes x 2 hops, first pass prefill.
    assert phases == ["prefill"] * 2 + ["decode"] * 4


# --- KV cache in generate() (issue #24) ---


def test_generate_defaults_to_using_the_cache(served_chain, input_ids):
    with ChainRunner(served_chain) as runner:
        _, passes = runner.generate(input_ids, max_new_tokens=3)

    # Decode hops send one token; prefill sends the whole prompt. If caching
    # were silently off, every hop's sent_bytes would be identical to the first.
    decode_hop0_bytes = [p.hops[0].sent_bytes for p in passes[1:]]
    assert all(b == 8 for b in decode_hop0_bytes)  # one int64 token id
    assert passes[0].hops[0].sent_bytes > decode_hop0_bytes[0]


def test_generate_with_cache_disabled_matches_the_reference_model_too(
    tiny_model, served_chain, input_ids
):
    # The explicit control path: no server-side state, every step re-sends the
    # whole growing prefix. Slower and more bandwidth, but must still be
    # correct -- it is what someone reaches for to isolate whether a bug is in
    # caching or elsewhere in the chain.
    with ChainRunner(served_chain) as runner:
        ids, _ = runner.generate(input_ids, max_new_tokens=3, use_cache=False)

    expected = input_ids
    with torch.inference_mode():
        for _ in range(3):
            nxt = tiny_model(input_ids=expected).logits[:, -1, :].argmax(-1, keepdim=True)
            expected = torch.cat([expected, nxt], dim=1)

    assert torch.equal(ids, expected)


def test_generate_with_cache_disabled_resends_the_whole_prefix_every_step(served_chain, input_ids):
    with ChainRunner(served_chain) as runner:
        _, passes = runner.generate(input_ids, max_new_tokens=3, use_cache=False)

    sent_lengths = [p.hops[0].sent_bytes // 8 for p in passes]  # int64 token count
    prompt_len = input_ids.shape[1]
    assert sent_lengths == [prompt_len, prompt_len + 1, prompt_len + 2]


def test_cached_and_uncached_generation_agree_on_the_same_tokens(served_chain, input_ids):
    # The two code paths compute the same thing through a different amount of
    # redundant recomputation; they must not be able to silently diverge.
    with ChainRunner(served_chain) as runner:
        cached_ids, _ = runner.generate(input_ids, max_new_tokens=4, use_cache=True)
    with ChainRunner(served_chain) as runner:
        uncached_ids, _ = runner.generate(input_ids, max_new_tokens=4, use_cache=False)

    assert torch.equal(cached_ids, uncached_ids)


def test_a_second_generation_after_the_first_completes_is_unaffected(
    tiny_model, served_chain, input_ids
):
    # end_of_request on generate()'s last step must actually free the session,
    # or a second, unrelated generation on the same chain would silently
    # inherit leftover cache state from the first.
    with ChainRunner(served_chain) as runner:
        runner.generate(input_ids, max_new_tokens=2)

        torch.manual_seed(123)
        second_prompt = torch.randint(0, 256, (1, 4))
        second_ids, _ = runner.generate(second_prompt, max_new_tokens=2)

    expected = second_prompt
    with torch.inference_mode():
        for _ in range(2):
            nxt = tiny_model(input_ids=expected).logits[:, -1, :].argmax(-1, keepdim=True)
            expected = torch.cat([expected, nxt], dim=1)

    assert torch.equal(second_ids, expected)


def test_generate_advances_the_expected_cache_position_each_step(served_chain, input_ids):
    """A full cached generation must satisfy the position check at every step.

    If ChainRunner.generate() miscounted -- forgetting that prefill advances the
    cache by the whole prompt, say -- every step after the first would be
    refused. This asserts the counting is right by the generation simply
    completing with the check armed.
    """
    with ChainRunner(served_chain) as runner:
        ids, passes = runner.generate(input_ids, max_new_tokens=4)

    assert ids.shape == (1, input_ids.shape[1] + 4)
    # The cache ends at prompt + (steps - 1), not prompt + steps: the prefill
    # caches the whole prompt, each later step feeds back the *previous* token,
    # and the final generated token is returned without ever being sent back in.
    final_hop = passes[-1].hops[0]
    assert final_hop.cache_length == input_ids.shape[1] + 3


def test_uncached_generation_does_not_arm_the_position_check(served_chain, input_ids):
    # With use_cache=False there is no session to be out of step with, and
    # sending a position expectation would be meaningless.
    with ChainRunner(served_chain) as runner:
        ids, passes = runner.generate(input_ids, max_new_tokens=2, use_cache=False)

    assert ids.shape == (1, input_ids.shape[1] + 2)
    assert all(hop.cache_length == 0 for result in passes for hop in result.hops)
