"""HTTP layer tests (issue #26).

Runs real shard servers in-process and drives them through the real FastAPI
app, so these exercise the whole stack: HTTP -> ChainRunner -> gRPC -> shard.
The only stand-in is the tokenizer, because the tiny randomly-initialised model
the suite uses has no tokenizer published anywhere and fetching a real one would
make the suite need network access.
"""

import pytest
from fastapi.testclient import TestClient

from torrent_llm.api import create_app
from torrent_llm.codec import get_codec
from torrent_llm.runner import TopologyConfig
from torrent_llm.shard import ShardRuntime
from torrent_llm.transport import serve


class FakeTokenizer:
    """Maps characters to ids and back.

    Deliberately trivial and reversible: these tests are about the HTTP layer
    and the chain beneath it, so a tokenizer with real subword behaviour would
    add a moving part without testing anything extra. Ids stay inside the tiny
    model's 256-token vocabulary.
    """

    def encode(self, text: str) -> list[int]:
        return [ord(c) % 256 for c in text]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(chr(int(i) % 256) for i in ids)


@pytest.fixture
def client(model_factory, num_layers):
    """A TestClient wired to a real 2-shard chain running in this process."""
    planning = TopologyConfig.from_dict(
        {
            "model_id": "tiny",
            "num_layers": num_layers,
            "nodes": [{"address": "127.0.0.1:0"}, {"address": "127.0.0.1:0"}],
        }
    )

    servers, addresses = [], []
    for spec in planning.shard_plan():
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

    with TestClient(create_app(config, FakeTokenizer())) as test_client:
        yield test_client

    for server in servers:
        server.stop(grace=None)


# --- health and topology ---


def test_health_reports_the_api_itself(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_topology_reports_what_each_node_hosts(client, num_layers):
    response = client.get("/topology")

    assert response.status_code == 200
    body = response.json()
    assert body["num_shards"] == 2
    assert body["num_layers"] == num_layers
    assert [s["layers"] for s in body["shards"]] == [
        f"[0:{num_layers // 2})",
        f"[{num_layers // 2}:{num_layers})",
    ]


def test_topology_asks_the_nodes_rather_than_echoing_the_config(client):
    # Each row carries the hidden size the *node* reported, which the config
    # never states -- proof this went over the wire rather than being read back
    # out of the topology file.
    body = client.get("/topology").json()

    assert all(s["hidden_size"] == 64 for s in body["shards"])


def test_topology_is_unavailable_rather_than_wrong_when_a_shard_is_down(model_factory):
    # Points at a port nothing is listening on. The failure has to be a clear
    # 503, not a hang or a plausible-looking empty topology.
    config = TopologyConfig.from_dict(
        {"model_id": "tiny", "num_layers": 6, "nodes": [{"address": "127.0.0.1:1"}]}
    )
    with TestClient(create_app(config, FakeTokenizer())) as client:
        response = client.get("/topology")

    assert response.status_code == 503
    assert "could not reach" in response.json()["detail"]


# --- generation ---


def test_generate_returns_completion_and_per_hop_metrics(client):
    response = client.post("/generate", json={"prompt": "hello", "max_new_tokens": 3})

    assert response.status_code == 200
    body = response.json()
    assert body["tokens_generated"] == 3
    assert body["full_text"].startswith("hello")
    # 3 generation steps x 2 hops.
    assert len(body["hops"]) == 6


def test_generate_separates_the_prompt_from_the_completion(client):
    body = client.post("/generate", json={"prompt": "abc", "max_new_tokens": 2}).json()

    assert body["prompt"] == "abc"
    assert not body["completion"].startswith("abc")
    assert body["full_text"] == "abc" + body["completion"]


def test_generate_labels_the_first_pass_prefill_and_the_rest_decode(client):
    body = client.post("/generate", json={"prompt": "hello", "max_new_tokens": 3}).json()

    phases = [h["phase"] for h in body["hops"]]
    assert phases == ["prefill"] * 2 + ["decode"] * 4


def test_generate_reports_durations_in_milliseconds(client):
    body = client.post("/generate", json={"prompt": "hi", "max_new_tokens": 2}).json()

    # Nanoseconds would put these in the millions; a sane millisecond figure for
    # a tiny model on loopback is well under a second per hop.
    for hop in body["hops"]:
        assert 0 < hop["wall_ms"] < 1000
        assert hop["transport_ms"] == pytest.approx(
            max(0.0, hop["wall_ms"] - hop["compute_ms"]), abs=1e-6
        )


def test_cached_decode_hops_send_a_single_token(client):
    body = client.post("/generate", json={"prompt": "hello world", "max_new_tokens": 3}).json()

    prefill_hop0 = next(h for h in body["hops"] if h["phase"] == "prefill" and h["hop"] == 0)
    decode_hop0 = [h for h in body["hops"] if h["phase"] == "decode" and h["hop"] == 0]

    assert all(h["sent_bytes"] == 8 for h in decode_hop0)  # one int64 token id
    assert prefill_hop0["sent_bytes"] > decode_hop0[0]["sent_bytes"]


def test_uncached_generation_resends_the_growing_prefix(client):
    body = client.post(
        "/generate",
        json={"prompt": "hello", "max_new_tokens": 3, "use_cache": False},
    ).json()

    hop0_tokens = [h["sent_bytes"] // 8 for h in body["hops"] if h["hop"] == 0]
    assert hop0_tokens == [5, 6, 7]  # "hello" is 5 characters under FakeTokenizer


def test_hop_0_reports_no_hidden_size_because_it_carries_token_ids(client):
    body = client.post("/generate", json={"prompt": "hi", "max_new_tokens": 1}).json()

    hop0 = next(h for h in body["hops"] if h["hop"] == 0)
    hop1 = next(h for h in body["hops"] if h["hop"] == 1)
    assert hop0["hidden_size"] == 0
    assert hop1["hidden_size"] == 64


def test_empty_prompt_is_rejected_with_a_useful_message(client):
    response = client.post("/generate", json={"prompt": "", "max_new_tokens": 2})

    assert response.status_code == 422
    assert "zero tokens" in response.json()["detail"]


def test_max_new_tokens_is_bounded(client):
    response = client.post("/generate", json={"prompt": "hi", "max_new_tokens": 0})
    assert response.status_code == 422

    response = client.post("/generate", json={"prompt": "hi", "max_new_tokens": 9999})
    assert response.status_code == 422


# --- the issue #24 payoff, as a single request ---


def test_compare_cache_shows_the_bandwidth_reduction(client):
    response = client.post("/compare-cache", json={"prompt": "hello", "max_new_tokens": 4})

    assert response.status_code == 200
    body = response.json()
    assert body["cached"]["total_sent_bytes"] < body["uncached"]["total_sent_bytes"]
    assert body["bandwidth_reduction"] > 1.0


def test_compare_cache_confirms_both_paths_produce_identical_text(client):
    body = client.post("/compare-cache", json={"prompt": "hello", "max_new_tokens": 4}).json()

    # The two differ only in how much they recompute, never in what they
    # compute. This field existing at all is what turns that from an assumption
    # into something the API checks on every call.
    assert body["same_output"] is True
    assert body["cached"]["full_text"] == body["uncached"]["full_text"]


# --- profiling ---


def test_profile_summarises_each_requested_context_length(client):
    response = client.post("/profile", json={"seq_lens": [8, 16], "repeats": 2})

    assert response.status_code == 200
    body = response.json()
    assert [p["seq_len"] for p in body["profiles"]] == [8, 16]
    for profile in body["profiles"]:
        assert len(profile["hops"]) == 2
        assert all(h["samples"] == 2 for h in profile["hops"])


def test_profile_states_the_regime_in_words(client):
    body = client.post("/profile", json={"seq_lens": [8], "repeats": 1}).json()

    verdict = body["profiles"][0]["verdict"]
    assert any(word in verdict for word in ("bandwidth-bound", "compute-bound", "mixed"))


def test_profile_payload_grows_with_context_length(client):
    body = client.post("/profile", json={"seq_lens": [8, 32], "repeats": 1}).json()

    def hop1_bytes(profile):
        return next(h["median_sent_bytes"] for h in profile["hops"] if h["hop"] == 1)

    short, long = body["profiles"]
    # Activation is seq x hidden x dtype_bytes, so 4x the context is 4x the wire.
    assert hop1_bytes(long) == pytest.approx(4 * hop1_bytes(short))


def test_profile_requires_at_least_one_context_length(client):
    response = client.post("/profile", json={"seq_lens": [], "repeats": 1})

    assert response.status_code == 422


def test_concurrent_requests_do_not_interleave_their_hop_records(client):
    """Each request must see only its own hops.

    A single shared profiler across requests would mix two runs' records into
    both responses, and the symptom would be implausible numbers rather than an
    error -- so this pins the per-request isolation down explicitly.
    """
    import concurrent.futures

    def run(tokens: int):
        return client.post("/generate", json={"prompt": "hello", "max_new_tokens": tokens}).json()

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run, [2, 3, 4, 5]))

    for expected_tokens, body in zip([2, 3, 4, 5], results, strict=True):
        assert body["tokens_generated"] == expected_tokens
        assert len(body["hops"]) == expected_tokens * 2
