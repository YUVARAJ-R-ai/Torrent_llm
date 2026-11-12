"""Peer-to-peer forwarding (#37): stage hands to stage, the last one delivers the token."""

from __future__ import annotations

import time

import pytest
import torch

from torrent_llm.net.identity import Identity
from torrent_llm.net.node import Node, NodeConfig
from torrent_llm.net.rpc import RpcError
from torrent_llm.pool.config import PoolConfig
from torrent_llm.pool.forwarding import RELAY, Forwarder, RelayChain
from torrent_llm.pool.hosting import OVERLAY, ShardHost
from torrent_llm.pool.overlay import serve_shards
from torrent_llm.runner import HopError
from torrent_llm.sampling import GREEDY
from torrent_llm.shard import ShardRuntime

STEPS = 5


def _node() -> Node:
    node = Node(
        NodeConfig(
            identity=Identity.generate(), listen_host="127.0.0.1", ping_interval=0.3, dead_after=3
        )
    )
    node.start()
    return node


class Swarm:
    """Overlay nodes on loopback; each may hold stages of a 3-stage tiny model."""

    def __init__(self, model_factory, num_layers: int, names: str) -> None:
        self.config = PoolConfig(model_id="tiny", num_layers=num_layers, stages=3)
        self.codecs = self.config.codec_set()
        self.codec = self.codecs.codec(self.config.policy()["fixed"])
        self.nodes = {name: _node() for name in names}
        for a in self.nodes.values():
            for b in self.nodes.values():
                if a is not b:
                    a.add_record(b.record())
        self._factory = model_factory
        self.hosts: dict[str, ShardHost] = {}
        self.forwarders: list[Forwarder] = []

    def hold(self, name: str, *stages: int) -> None:
        host = ShardHost(
            self.config.specs(),
            lambda spec: ShardRuntime(self._factory(), spec),
            model_id="tiny",
            transport=OVERLAY,
            codecs=self.codecs,
        )
        for s in stages:
            host.start(s)
        deadline = time.monotonic() + 30
        while not all(h.status == "ready" for h in host.hosting()):
            assert time.monotonic() < deadline, "stages did not load"
            time.sleep(0.05)
        self.hosts[name] = host
        serve_shards(self.nodes[name], host)  # requester routing, for comparison
        self.forwarders.append(Forwarder(self.nodes[name], host))

    def chain(self, asker: str, route: list[tuple[str, int]], **kw) -> RelayChain:
        return RelayChain(
            self.nodes[asker],
            [(self.nodes[n].peer_id, s, n) for n, s in route],
            self.codec,
            codecs=self.codecs,
            **kw,
        )

    def close(self) -> None:
        for f in self.forwarders:
            f.close()
        for h in self.hosts.values():
            h.close()
        for n in self.nodes.values():
            n.close()


@pytest.fixture
def swarm(model_factory, num_layers):
    made: list[Swarm] = []

    def make(names: str) -> Swarm:
        made.append(Swarm(model_factory, num_layers, names))
        return made[-1]

    yield make
    for s in made:
        s.close()


def _reference(model, prompt: torch.Tensor, steps: int) -> list[int]:
    ids, out = prompt, []
    with torch.no_grad():
        for _ in range(steps):
            token = model(ids).logits[:, -1].argmax(-1, keepdim=True)
            out.append(int(token))
            ids = torch.cat([ids, token], dim=1)
    return out


def _decode(chain: RelayChain, prompt: torch.Tensor, steps: int, rid: str = "r1"):
    sent, position, tokens, passes = prompt, 0, [], []
    for step in range(steps):
        result = chain.forward(
            sent,
            request_id=rid,
            use_cache=True,
            end_of_request=step == steps - 1,
            expected_cache_position=position,
            sampling=GREEDY.for_step(step),
        )
        passes.append(result)
        position += sent.shape[1]
        sent = result.token_ids
        tokens.append(int(sent))
    return tokens, passes


PROMPT = torch.tensor([[1, 17, 42, 99, 7, 3]])


def test_three_relaying_peers_match_the_whole_model(swarm, tiny_model):
    s = swarm("abcd")
    for name, stage in (("b", 0), ("c", 1), ("d", 2)):
        s.hold(name, stage)
    chain = s.chain("a", [("b", 0), ("c", 1), ("d", 2)])
    tokens, passes = _decode(chain, PROMPT, STEPS)
    assert tokens == _reference(tiny_model, PROMPT, STEPS)

    last = passes[-1]
    assert [h.hop for h in last.hops] == [0, 1, 2]
    assert all(h.compute_ns > 0 for h in last.hops)  # per-peer compute from the trace
    # Only a token comes home: one int64, not hidden states or logits.
    assert last.hops[-1].received_bytes == 8
    # A decode step carries one position on every link.
    assert last.hops[1].sent_numel == last.hops[2].sent_numel == 64


def test_the_asker_can_hold_stages_itself(swarm, tiny_model):
    s = swarm("ab")
    s.hold("a", 0, 2)  # the privacy-mode shape: first and last stage at home
    s.hold("b", 1)
    tokens, _ = _decode(s.chain("a", [("a", 0), ("b", 1), ("a", 2)]), PROMPT, STEPS)
    assert tokens == _reference(tiny_model, PROMPT, STEPS)

    s2 = swarm("xy")
    s2.hold("x", 1)
    s2.hold("y", 0, 2)
    tokens, _ = _decode(s2.chain("x", [("y", 0), ("x", 1), ("y", 2)]), PROMPT, STEPS)
    assert tokens == _reference(tiny_model, PROMPT, STEPS)


def test_a_wrong_cache_position_is_refused_on_the_hop_that_sees_it(swarm):
    s = swarm("abcd")
    for name, stage in (("b", 0), ("c", 1), ("d", 2)):
        s.hold(name, stage)
    chain = s.chain("a", [("b", 0), ("c", 1), ("d", 2)])
    _decode(chain, PROMPT, 1, rid="r2")
    with pytest.raises(HopError) as caught:
        chain.forward(
            torch.tensor([[5]]),
            request_id="r2",
            use_cache=True,
            expected_cache_position=999,
            sampling=GREEDY,
        )
    assert caught.value.hop == 0
    assert "cache_lost" in str(caught.value)


def test_an_unreachable_next_stage_is_named(swarm):
    s = swarm("abcd")
    for name, stage in (("b", 0), ("c", 1), ("d", 2)):
        s.hold(name, stage)
    chain = s.chain("a", [("b", 0), ("c", 1), ("d", 2)])
    s.nodes["d"].close()
    with pytest.raises(HopError) as caught:
        _decode(chain, PROMPT, 1)
    assert caught.value.hop == 2
    assert caught.value.address == "d"
    assert caught.value.is_node_loss


def test_a_stage_that_goes_silent_is_named_at_the_deadline(swarm, monkeypatch):
    s = swarm("abcd")
    for name, stage in (("b", 0), ("c", 1), ("d", 2)):
        s.hold(name, stage)
    host = s.hosts["d"]
    real = host.forward
    monkeypatch.setattr(host, "forward", lambda stage, req: (time.sleep(3), real(stage, req))[1])
    chain = s.chain("a", [("b", 0), ("c", 1), ("d", 2)], deadline_s=1.0)
    t0 = time.monotonic()
    with pytest.raises(HopError) as caught:
        _decode(chain, PROMPT, 1)
    assert time.monotonic() - t0 < 3
    assert caught.value.hop == 2
    assert s.nodes["d"].peer_id[:8] in str(caught.value)


def test_only_the_previous_hop_may_send_a_hop(swarm):
    s = swarm("abcd")
    s.hold("c", 1)
    route = [[s.nodes[n].peer_id, i] for i, n in enumerate("bcd")]
    header = {
        "rid": "x",
        "seq": 0,
        "hop": 1,
        "route": route,
        "reply_to": s.nodes["a"].peer_id,
        "trace": [{}],
    }
    with pytest.raises(RpcError) as caught:  # a is not b, the holder of hop 0
        s.nodes["a"].call(s.nodes["c"].peer_id, RELAY, header, b"", timeout=5)
    assert caught.value.code == "denied"
    header["hop"] = 0
    header["trace"] = []
    with pytest.raises(RpcError) as caught:  # hop 0 is b's, not c's
        s.nodes["a"].call(s.nodes["c"].peer_id, RELAY, header, b"", timeout=5)
    assert caught.value.code == "denied"
    header["route"] = [["not-a-peer", 0]]
    with pytest.raises(RpcError) as caught:
        s.nodes["a"].call(s.nodes["c"].peer_id, RELAY, header, b"", timeout=5)
    assert caught.value.code == "bad_request"


def _table(s: Swarm, me: str, holders: dict[str, tuple[int, ...]], speed: dict[str, float]):
    from torrent_llm.pool.members import ACTIVE, READY, HostedStage, Member, MemberTable

    def member(name: str) -> Member:
        return Member(
            name=s.nodes[name].peer_id,
            host="h",
            control_port=1,
            slots=3,
            speed=speed.get(name, 1.0),
            state=ACTIVE,
            hosting=tuple(
                HostedStage(stage=x, port=0, status=READY) for x in holders.get(name, ())
            ),
            heartbeat=1,
        )

    table = MemberTable(s.nodes[me].peer_id)
    table.set_self(member(me))
    table.merge([member(n) for n in s.nodes if n != me])
    return table


def test_the_pool_runner_relays_and_fails_over(swarm, tiny_model):
    import dataclasses

    from torrent_llm.pool.runner import PoolRunner

    s = swarm("abcde")
    holders = {"b": (0,), "c": (1,), "e": (1,), "d": (2,)}
    for name, stages in holders.items():
        s.hold(name, *stages)
    want = _reference(tiny_model, PROMPT, STEPS)

    for routing in ("requester", "relay"):
        config = dataclasses.replace(s.config, routing=routing)
        table = _table(s, "a", holders, {})
        runner = PoolRunner(config, table, me=s.nodes["a"].peer_id, node=s.nodes["a"])
        result = runner.generate(PROMPT, max_new_tokens=STEPS)
        assert result.ids[0, PROMPT.shape[1] :].tolist() == want, routing
        assert result.failovers == 0

    # c is the faster copy of stage 1, so it is chosen first; it is gone, so the
    # request must move to e and still give the same answer.
    s.nodes["c"].close()
    config = dataclasses.replace(s.config, routing="relay")
    table = _table(s, "a", holders, {"c": 10.0})
    runner = PoolRunner(config, table, me=s.nodes["a"].peer_id, node=s.nodes["a"])
    result = runner.generate(PROMPT, max_new_tokens=STEPS)
    assert result.ids[0, PROMPT.shape[1] :].tolist() == want
    assert result.failovers == 1
    assert result.events[0].member == s.nodes["c"].peer_id
