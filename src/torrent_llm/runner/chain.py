"""Driving a request through the whole shard chain.

The runner is the only component that knows the chain exists as a chain. Shards
know their own layers; clients know one hop. Sequencing lives here, which is
where rerouting on node failure (issue #16) will eventually go.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Collection
from dataclasses import dataclass

import grpc
import torch

from torrent_llm.codec import get_codec
from torrent_llm.profile import HopProfiler
from torrent_llm.runner.config import TopologyConfig
from torrent_llm.sampling import GREEDY, Sampling, select_tokens
from torrent_llm.transport import HopResult, ShardClient

logger = logging.getLogger(__name__)


class HopError(RuntimeError):
    """One hop of the chain failed, and which one.

    A plain ``grpc.RpcError`` says a call failed but not which node of the chain
    it was. Failover needs exactly that: it has to know which replica to stop
    using. The original error is kept as ``__cause__`` and its status in ``code``.
    """

    def __init__(self, hop: int, address: str, cause: grpc.RpcError) -> None:
        code = cause.code() if hasattr(cause, "code") else None
        detail = cause.details() if hasattr(cause, "details") else str(cause)
        super().__init__(f"hop {hop} ({address}) failed: {code.name if code else '?'}: {detail}")
        self.hop = hop
        self.address = address
        self.code = code

    @property
    def is_node_loss(self) -> bool:
        """True when the node itself is gone or unreachable, not merely unhappy.

        ``FAILED_PRECONDITION`` (a lost cache session) is not node loss: the node
        answered. It still needs a re-prefill, but it is worth trying again.
        """
        return self.code in (
            grpc.StatusCode.UNAVAILABLE,
            grpc.StatusCode.DEADLINE_EXCEEDED,
            grpc.StatusCode.CANCELLED,
        )


@dataclass
class ChainResult:
    """Output of one pass through the chain, plus per-hop measurements.

    Exactly one of ``logits`` and ``token_ids`` is set: ``token_ids`` when the
    pass asked the final shard to choose the next token (issue #33), ``logits``
    otherwise.
    """

    logits: torch.Tensor | None
    hops: list[HopResult]
    request_id: str
    #: ``(batch, 1)`` ids chosen by the final shard, when the pass sampled there.
    token_ids: torch.Tensor | None = None

    @property
    def total_wall_ns(self) -> int:
        return sum(h.wall_ns for h in self.hops)

    @property
    def total_compute_ns(self) -> int:
        return sum(h.compute_ns for h in self.hops)

    @property
    def total_transport_ns(self) -> int:
        """End-to-end latency not explained by shard compute."""
        return sum(h.transport_ns for h in self.hops)

    @property
    def total_sent_bytes(self) -> int:
        """Every byte this request put on the wire, across all hops."""
        return sum(h.sent_bytes for h in self.hops)


class ChainRunner:
    """Walks token ids through every shard and returns logits."""

    def __init__(
        self,
        config: TopologyConfig,
        *,
        profiler: HopProfiler | None = None,
        token: str | None = None,
    ) -> None:
        self.config = config
        self.profiler = profiler
        self.clients = [
            ShardClient(
                node.address,
                get_codec(config.codec, **config.codec_args),
                link=node.link or config.link,
                token=token,
            )
            for node in config.nodes
        ]

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
        phase: str = "prefill",
        logits_keep_last: int = 0,
        request_id: str | None = None,
        use_cache: bool = False,
        end_of_request: bool = False,
        expected_cache_position: int | None = None,
        sampling: Sampling | None = None,
    ) -> ChainResult:
        """One full forward pass over the chain.

        Every hop is handed to the profiler, if one is attached, with the shape
        of what was actually sent — token ids on hop 0, activations after that.

        Args:
            logits_keep_last: Trailing logit positions to return; ``0`` means
                all. Greedy decoding needs only ``1``, and asking for all of
                them puts a batch x seq x vocab tensor on the wire, which on a
                modern tokenizer is far larger than any activation in the chain.
            request_id: Correlates this call with earlier ones under the same
                KV cache session (issue #24). Only meaningful together with
                ``use_cache=True``; a fresh id is generated when omitted, which
                is correct for a one-off, uncached call.
            use_cache: Reuse (or start, on the first call for this
                ``request_id``) a server-side KV cache on every shard, so
                ``input_ids``/``hidden_states`` only need to carry the new
                positions rather than the whole sequence. See
                :meth:`generate` for the loop that actually exploits this —
                calling ``forward`` directly with caching on is mostly useful
                for building a custom decode loop.
            end_of_request: Tell every shard this is the last call for
                ``request_id``, so each frees its cached state immediately. See
                the ``end_of_request`` field in ``activation.proto`` for why
                this exists alongside a TTL-based backstop rather than instead
                of one.
            expected_cache_position: How many positions each shard's cache
                should already hold (issue #28). Every shard in the chain sees
                the same value, because they advance in lockstep -- shard 3
                having processed a different number of positions than shard 0
                is itself the bug this catches.
            sampling: Have the final shard pick the next token by this rule and
                return only its id (issue #33). The result then carries
                ``token_ids`` instead of ``logits``. Leave it out wherever the
                logits themselves are the point, as in perplexity evaluation.
        """
        request_id = request_id or uuid.uuid4().hex
        last = len(self.clients) - 1
        hops: list[HopResult] = []

        sent = input_ids
        result = self._hop(
            0,
            input_ids,
            request_id=request_id,
            is_token_ids=True,
            position_ids=position_ids,
            use_cache=use_cache,
            end_of_request=end_of_request,
            expected_cache_position=expected_cache_position,
            sampling=sampling if last == 0 else None,
        )
        hops.append(result)
        self._profile(result, request_id, 0, sent, phase)

        for hop in range(1, len(self.clients)):
            sent = result.tensor
            result = self._hop(
                hop,
                sent,
                request_id=request_id,
                position_ids=position_ids,
                logits_keep_last=logits_keep_last,
                use_cache=use_cache,
                end_of_request=end_of_request,
                expected_cache_position=expected_cache_position,
                # Only the shard that owns the LM head reads it, so it rides on
                # that hop alone.
                sampling=sampling if hop == last else None,
            )
            hops.append(result)
            self._profile(result, request_id, hop, sent, phase)

        if not result.is_final:
            raise RuntimeError(
                "chain ended without reaching the shard that owns the LM head; "
                "the topology is probably missing its tail shard"
            )
        if sampling is not None:
            return ChainResult(
                logits=None, token_ids=result.tensor, hops=hops, request_id=request_id
            )
        return ChainResult(logits=result.tensor, hops=hops, request_id=request_id)

    def _hop(self, hop: int, tensor: torch.Tensor, **kwargs) -> HopResult:
        """Send one hop, naming the hop if it fails."""
        client = self.clients[hop]
        try:
            return client.forward(tensor, hop=hop, **kwargs)
        except grpc.RpcError as exc:
            raise HopError(hop, client.address, exc) from exc

    def _profile(
        self,
        result: HopResult,
        request_id: str,
        hop: int,
        sent: torch.Tensor,
        phase: str,
    ) -> None:
        if self.profiler is None:
            return
        self.profiler.record(
            result,
            request_id=request_id,
            address=self.clients[hop].address,
            shape=tuple(sent.shape),
            dtype=str(sent.dtype).removeprefix("torch."),
            phase=phase,
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int = 16,
        use_cache: bool = True,
        stop_token_ids: Collection[int] = (),
        sampling: Sampling = GREEDY,
        sample_on_shard: bool = True,
    ) -> tuple[torch.Tensor, list[ChainResult]]:
        """Decode, greedily unless ``sampling`` says otherwise.

        With ``use_cache=True`` (the default, issue #24), every shard keeps a
        KV cache for this generation's ``request_id``: the first step sends the
        whole prompt, and every step after that sends only the single newly
        generated token. Per-step wire payloads are therefore real cached-decode
        payloads — a few KiB, not a repeated prefill — which is what makes
        decode-regime bandwidth something this project can actually measure
        instead of only projecting analytically (see docs/bandwidth-regimes.md).

        ``use_cache=False`` keeps the original behaviour: every step re-runs the
        whole growing prefix through every shard with no server-side state at
        all. Slower and far more bandwidth, but it is still there as an explicit
        control — the cleanest way to *measure* the cache's payoff is to run
        both and diff the profiler output, not to trust that caching helped.

        Cleanup on the happy path is automatic: the last step is sent with
        ``end_of_request=True``, so every shard drops its cache for this request
        as soon as generation finishes. If this method raises partway through
        (a shard error, a dropped connection), that signal never goes out and
        the abandoned sessions are freed later by each shard's own TTL sweep
        rather than by anything this method does — see ``SessionStore`` for why
        a best-effort signal plus a backstop beats trying to guarantee cleanup
        from the calling side, which a network partition can defeat anyway.

        ``stop_token_ids`` ends generation as soon as the model picks one of
        them, the way a chat model says it has finished answering. The stop
        token itself is not appended. Stopping early has the same cleanup
        story as a failure: the step that chose the stop token was not marked
        ``end_of_request`` (nothing could know yet), so each shard frees that
        session on its TTL sweep. Sending one more pass only to free a few KiB
        of cache would cost a full round trip per request.

        With ``sample_on_shard=True`` (the default, issue #33) the final shard
        applies ``sampling`` and returns only the chosen token, so the reply on
        the last hop is 8 bytes instead of ``vocab_size x 4``. ``False`` keeps the
        original path -- the shard returns the last position's logits and the
        token is chosen here -- as an explicit control. Both paths make the same
        choice from the same logits; ``sampling`` is reseeded per step either way.
        """
        request_id = uuid.uuid4().hex
        ids = input_ids
        # What gets sent *this* step: the whole prompt once, then one token at a
        # time once the cache is carrying the rest of the context.
        sent = input_ids
        # How many positions every shard's cache should already hold before this
        # step. Starts at 0 (nothing cached yet) and advances by whatever was
        # sent. Checking it is what turns a lost session into an error instead
        # of silently wrong output -- see issue #28.
        cache_position = 0
        passes: list[ChainResult] = []
        for step in range(max_new_tokens):
            is_last_step = step == max_new_tokens - 1
            result = self.forward(
                sent,
                phase="prefill" if step == 0 else "decode",
                logits_keep_last=1,
                request_id=request_id,
                use_cache=use_cache,
                end_of_request=use_cache and is_last_step,
                expected_cache_position=cache_position if use_cache else None,
                sampling=sampling.for_step(step) if sample_on_shard else None,
            )
            passes.append(result)
            cache_position += sent.shape[1]
            if sample_on_shard:
                assert result.token_ids is not None
                next_token = result.token_ids
            else:
                assert result.logits is not None
                next_token = select_tokens(result.logits, sampling.for_step(step))
            next_token = next_token.to(ids.device)
            if next_token.item() in stop_token_ids:
                break
            ids = torch.cat([ids, next_token], dim=1)
            sent = next_token if use_cache else ids
        return ids, passes

    def describe(self) -> list[dict[str, object]]:
        """What every node reports hosting. Verifies the chain before a real run."""
        out = []
        for client in self.clients:
            info = client.info()
            out.append(
                {
                    "address": client.address,
                    "shard": info.shard_index,
                    "layers": f"[{info.layer_start}:{info.layer_end})",
                    "hidden_size": info.hidden_size,
                    "device": info.device,
                    "dtype": info.dtype,
                    "model_id": info.model_id,
                    "hostname": info.hostname,
                    "os": info.os,
                }
            )
        return out

    def close(self) -> None:
        for client in self.clients:
            client.close()

    def __enter__(self) -> ChainRunner:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
