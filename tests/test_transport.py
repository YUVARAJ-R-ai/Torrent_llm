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


# --- KV cache over the wire (issue #24) ---
#
# The in-process case (tests/test_shard_runtime.py) already proves the math.
# What is specific to this layer is the *session lifecycle*: use_cache has to
# survive a real request/reply round trip and be reusable across separate gRPC
# calls, end_of_request has to actually free server memory, and a client that
# forgets use_cache has to fall back to the old stateless behaviour rather than
# silently doing something wrong.


def cached_generate(clients, prompt_ids, num_new_tokens, request_id="gen-1"):
    """Drive greedy decoding across the chain using the server-side cache."""

    def step(tensor, *, is_token_ids, is_last_call):
        result = clients[0].forward(
            tensor,
            hop=0,
            is_token_ids=is_token_ids,
            request_id=request_id,
            use_cache=True,
            end_of_request=is_last_call,
        )
        for hop, client in enumerate(clients[1:], start=1):
            result = client.forward(
                result.tensor,
                hop=hop,
                request_id=request_id,
                use_cache=True,
                end_of_request=is_last_call,
                logits_keep_last=1,
            )
        return result

    tokens = [prompt_ids]
    result = step(prompt_ids, is_token_ids=True, is_last_call=num_new_tokens == 1)
    next_token = result.tensor.argmax(dim=-1)
    tokens.append(next_token)

    for i in range(1, num_new_tokens):
        result = step(next_token, is_token_ids=True, is_last_call=(i == num_new_tokens - 1))
        next_token = result.tensor.argmax(dim=-1)
        tokens.append(next_token)

    return torch.cat(tokens, dim=1)


def test_cached_generation_over_grpc_matches_greedy_decode_on_the_reference_model(
    tiny_model, chain, input_ids
):
    with torch.inference_mode():
        expected = input_ids
        for _ in range(3):
            nxt = tiny_model(input_ids=expected).logits[:, -1:, :].argmax(dim=-1)
            expected = torch.cat([expected, nxt], dim=1)

    generated = cached_generate(chain, input_ids, num_new_tokens=3)

    assert torch.equal(generated, expected)


def test_decode_step_after_prefill_sends_a_single_token_not_the_whole_sequence(chain, input_ids):
    request_id = "gen-payload-size"

    def step(tensor, *, is_token_ids, is_last_call):
        # Both hops must be continued on every call, prefill included -- sending
        # only hop 0 and skipping hop 1 would leave hop 1's session stuck at the
        # prefill length while hop 0's moves on, which is a real way to corrupt
        # a session and not something this test is trying to exercise.
        r0 = chain[0].forward(
            tensor,
            hop=0,
            is_token_ids=is_token_ids,
            request_id=request_id,
            use_cache=True,
            end_of_request=is_last_call,
        )
        r1 = chain[1].forward(
            r0.tensor,
            hop=1,
            request_id=request_id,
            use_cache=True,
            end_of_request=is_last_call,
            logits_keep_last=1,
        )
        return r0, r1

    prefill_r0, prefill_r1 = step(input_ids, is_token_ids=True, is_last_call=False)
    next_token = prefill_r1.tensor.argmax(dim=-1)

    decode_r0, _ = step(next_token, is_token_ids=True, is_last_call=True)

    # This is the entire point of the cache: hop 0's decode-step payload is one
    # token, regardless of how long the prompt was.
    assert decode_r0.sent_bytes == 1 * 8  # one int64 token id
    assert decode_r0.sent_bytes < prefill_r0.sent_bytes


def test_starting_a_new_generation_with_a_reused_request_id_after_cleanup_is_correct(
    tiny_model, chain, input_ids
):
    # Runs one full generation to completion (freeing the session via
    # end_of_request on the last call), then runs a second, different
    # generation reusing the same request_id. If cleanup had not actually
    # happened, the second generation would silently continue the first one's
    # cache and diverge from the reference model.
    request_id = "reused-id"
    cached_generate(chain, input_ids, num_new_tokens=2, request_id=request_id)

    torch.manual_seed(99)
    second_prompt = torch.randint(0, 256, (1, 5))
    with torch.inference_mode():
        expected = second_prompt
        for _ in range(2):
            nxt = tiny_model(input_ids=expected).logits[:, -1:, :].argmax(dim=-1)
            expected = torch.cat([expected, nxt], dim=1)

    generated = cached_generate(chain, second_prompt, num_new_tokens=2, request_id=request_id)

    assert torch.equal(generated, expected)


def test_use_cache_false_is_unaffected_and_stays_the_documented_default(chain, input_ids):
    # Every test elsewhere in this file calls .forward() without touching
    # use_cache at all -- this just makes the default explicit as a contract.
    hops = walk(chain, input_ids)

    assert hops[-1].is_final


# --- lost cache sessions must fail loudly (issue #28) ---
#
# Before this check existed, a decode step sent to a shard that had lost the
# session built a fresh empty cache, derived position_ids = [0] from it, and ran
# a mid-generation token as if it were the first of a new sequence. Generation
# continued and the output was quietly wrong -- the worst possible shape for a
# failure in a swarm where nodes leaving mid-request is the normal case.


def test_a_lost_session_is_refused_rather_than_computed_from_zero(chain, input_ids):
    import grpc

    # A decode-shaped call under a request_id no shard has ever seen: exactly
    # what a restarted shard, an expired session, or a request landing on a
    # different replica looks like from the server's side.
    with pytest.raises(grpc.RpcError) as excinfo:
        chain[0].forward(
            torch.tensor([[42]]),
            hop=0,
            is_token_ids=True,
            request_id="never-prefilled",
            use_cache=True,
            expected_cache_position=6,
        )

    assert excinfo.value.code() == grpc.StatusCode.FAILED_PRECONDITION
    assert "cache holds 0 positions but the client expected 6" in excinfo.value.details()
    # The message has to say what to do, not just what was wrong.
    assert "Re-run the prefill" in excinfo.value.details()


def test_a_healthy_session_passes_the_check(chain, input_ids):
    request_id = "healthy"
    prefill = chain[0].forward(
        input_ids,
        hop=0,
        is_token_ids=True,
        request_id=request_id,
        use_cache=True,
        expected_cache_position=0,
    )

    assert prefill.cache_length == input_ids.shape[1]

    decode = chain[0].forward(
        torch.tensor([[42]]),
        hop=0,
        is_token_ids=True,
        request_id=request_id,
        use_cache=True,
        expected_cache_position=input_ids.shape[1],
    )

    assert decode.cache_length == input_ids.shape[1] + 1


def test_the_check_is_opt_in_so_a_caller_that_does_not_track_position_still_works(chain, input_ids):
    # Omitting expected_cache_position skips validation entirely. Presence, not
    # value, is what arms the check -- 0 is a legitimate "this is the prefill".
    result = chain[0].forward(
        input_ids, hop=0, is_token_ids=True, request_id="unchecked", use_cache=True
    )

    assert result.cache_length == input_ids.shape[1]


def test_uncached_requests_report_no_cache_length(chain, input_ids):
    result = chain[0].forward(input_ids, hop=0, is_token_ids=True, request_id="nocache")

    assert result.cache_length == 0


def test_binding_a_port_that_is_already_serving_fails_loudly():
    """A stale shard process must be a startup error, not a silent oddity.

    gRPC enables SO_REUSEPORT by default, which makes a duplicate bind ambiguous:
    it logs a failure and still returns a port. On a rig where shards get
    restarted by hand between experiments, that turns "did the old process
    actually die?" into something you diagnose from confusing results rather
    than from a crash at startup.
    """
    import grpc
    from transformers import AutoModelForCausalLM, LlamaConfig

    from torrent_llm.codec import get_codec
    from torrent_llm.shard import ShardRuntime, plan_even
    from torrent_llm.transport import serve

    def make_shard():
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(
            LlamaConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                vocab_size=64,
                max_position_embeddings=32,
            )
        ).eval()
        return ShardRuntime(model, plan_even(2, 1)[0])

    first, port = serve(make_shard(), get_codec("raw"), host="127.0.0.1")
    try:
        with pytest.raises((RuntimeError, grpc.RpcError)):
            serve(make_shard(), get_codec("raw"), host="127.0.0.1", port=port)
    finally:
        first.stop(grace=None)
