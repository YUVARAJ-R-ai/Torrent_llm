"""HTTP layer over the shard chain (issue #26).

Instrumentation, not a serving layer. No auth, no batching, no multi-tenancy --
this exists so experiments and the dashboard can drive the chain without a
one-off Python script each time, and scope creep here is time not spent on the
compressor.

**This process never loads a model.** It holds a tokenizer and a set of gRPC
clients; the weights live in the ``torrent-shard`` processes it talks to. That
separation is the whole point of the topology config, and it means the API can
be restarted freely without paying to reload a checkpoint.
"""

from __future__ import annotations

import logging
from typing import Protocol

import torch
from fastapi import FastAPI, HTTPException

from torrent_llm.api.schemas import (
    CacheComparisonResponse,
    GenerateRequest,
    GenerateResponse,
    HopMetrics,
    HopSummaryMetrics,
    ProfileRequest,
    ProfileResponse,
    SeqLenProfile,
    ShardInfo,
    TopologyResponse,
)
from torrent_llm.profile import HopProfiler, bandwidth_verdict, summarize_by_hop
from torrent_llm.runner import ChainRunner, TopologyConfig

logger = logging.getLogger(__name__)

#: Upper bound on synthetic token ids used for profiling. Well under the
#: smallest vocabulary any real tokenizer ships, so profiling never needs to
#: know the model's vocab size to avoid indexing past its embedding table.
_PROFILE_TOKEN_CEILING = 100


class Tokenizer(Protocol):
    """The only two tokenizer operations this layer needs.

    Declared as a Protocol rather than typed as ``PreTrainedTokenizer`` so tests
    can inject a trivial stand-in: the tiny randomly-initialised models the test
    suite uses have no tokenizer published anywhere, and pulling a real one from
    the Hub just to encode "hello" would make the suite need network access.
    """

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str: ...


def create_app(config: TopologyConfig, tokenizer: Tokenizer) -> FastAPI:
    """Build the API around an already-running chain.

    Args:
        config: Topology describing which address hosts which layers. The shard
            servers it names must already be up; this function does not start
            them and does not check that they are reachable, so that the API can
            boot in any order relative to the nodes.
        tokenizer: Anything satisfying :class:`Tokenizer`.
    """
    app = FastAPI(
        title="Torrent-LLM",
        description="Instrumentation over a layer-sharded inference chain.",
        version="0.1.0",
    )

    def runner_with_profiler() -> tuple[ChainRunner, HopProfiler]:
        """A fresh runner and profiler for one request.

        Deliberately per-request rather than a shared long-lived runner. A
        profiler accumulates records into a list, so a single shared one would
        interleave two concurrent requests' hops into the same result -- and the
        bug would look like implausible numbers rather than an error. Building a
        runner costs a few gRPC channels, which is nothing next to a forward
        pass, so correctness wins easily here.
        """
        profiler = HopProfiler()
        return ChainRunner(config, profiler=profiler), profiler

    def encode_prompt(prompt: str) -> torch.Tensor:
        ids = tokenizer.encode(prompt)
        if not ids:
            # An empty prompt gives the first shard a zero-length sequence,
            # which fails somewhere deep in attention with a shape error that
            # says nothing about the actual cause.
            raise HTTPException(
                status_code=422,
                detail="prompt encoded to zero tokens; nothing to run through the chain",
            )
        return torch.tensor([ids], dtype=torch.long)

    @app.get("/topology", response_model=TopologyResponse)
    def topology() -> TopologyResponse:
        """What each node reports hosting.

        Asks the nodes themselves rather than reading it off the config, so a
        node serving a different layer range than the config claims shows up
        here as a discrepancy instead of being silently papered over.
        """
        try:
            with ChainRunner(config) as runner:
                rows = runner.describe()
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail=f"could not reach every shard in the chain: {exc}"
            ) from exc

        return TopologyResponse(
            model_id=config.model_id,
            num_layers=config.num_layers,
            num_shards=len(config.nodes),
            codec=config.codec,
            dtype=config.dtype,
            shards=[ShardInfo(**row) for row in rows],
        )

    def _generate(request: GenerateRequest) -> GenerateResponse:
        """Shared by /generate and /compare-cache so the two cannot drift apart."""
        input_ids = encode_prompt(request.prompt)
        prompt_len = input_ids.shape[1]

        runner, profiler = runner_with_profiler()
        try:
            ids, passes = runner.generate(
                input_ids,
                max_new_tokens=request.max_new_tokens,
                use_cache=request.use_cache,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=502, detail=f"chain failed mid-generation: {exc}"
            ) from exc
        finally:
            runner.close()

        generated_ids = ids[0, prompt_len:].tolist()
        records = profiler.records

        return GenerateResponse(
            prompt=request.prompt,
            completion=tokenizer.decode(generated_ids, skip_special_tokens=True),
            full_text=tokenizer.decode(ids[0].tolist(), skip_special_tokens=True),
            tokens_generated=len(generated_ids),
            use_cache=request.use_cache,
            total_sent_bytes=sum(r.sent_bytes for r in records),
            total_wall_ms=sum(r.wall_ns for r in records) / 1e6,
            total_compute_ms=sum(r.compute_ns for r in records) / 1e6,
            total_transport_ms=sum(r.transport_ns for r in records) / 1e6,
            hops=[HopMetrics.from_record(r) for r in records],
        )

    @app.post("/generate", response_model=GenerateResponse)
    def generate(request: GenerateRequest) -> GenerateResponse:
        """Generate, and report what every hop cost."""
        return _generate(request)

    @app.post("/compare-cache", response_model=CacheComparisonResponse)
    def compare_cache(request: GenerateRequest) -> CacheComparisonResponse:
        """Run the same prompt with and without the KV cache.

        Both runs happen here rather than being two separate calls the caller
        pairs up, because the headline number is the ratio between them and a
        mispaired ratio is worse than no ratio at all.
        """
        cached = _generate(request.model_copy(update={"use_cache": True}))
        uncached = _generate(request.model_copy(update={"use_cache": False}))

        return CacheComparisonResponse(
            prompt=request.prompt,
            tokens_generated=cached.tokens_generated,
            cached=cached,
            uncached=uncached,
            bandwidth_reduction=(
                uncached.total_sent_bytes / cached.total_sent_bytes
                if cached.total_sent_bytes
                else 1.0
            ),
            # Greedy decoding is deterministic, and the two paths differ only in
            # how much they recompute. Identical text is the invariant; this
            # surfaces a violation rather than leaving it to be noticed later.
            same_output=cached.full_text == uncached.full_text,
        )

    @app.post("/profile", response_model=ProfileResponse)
    def profile(request: ProfileRequest) -> ProfileResponse:
        """Run a prefill sweep and summarise each context length.

        Prefill only. Mixing decode records into these summaries would average
        two regimes whose payloads differ by three orders of magnitude, which is
        exactly the mistake docs/bandwidth-regimes.md exists to prevent.
        """
        runner, profiler = runner_with_profiler()
        try:
            for seq_len in request.seq_lens:
                # Random ids rather than a real prompt: this measures transport
                # and compute against sequence length, and what the tokens
                # actually say has no bearing on either.
                #
                # The upper bound is deliberately far below any real
                # tokenizer's vocabulary. This layer does not know the model's
                # vocab size -- that lives in the shard processes -- and an id
                # past the end of the embedding table fails with a bare
                # "index out of range in self" from deep inside torch, naming
                # nothing useful. Staying in a range every model has avoids
                # needing to know.
                ids = torch.randint(0, _PROFILE_TOKEN_CEILING, (1, seq_len))
                for _ in range(request.repeats):
                    runner.forward(ids, logits_keep_last=1)
        except Exception as exc:
            raise HTTPException(
                status_code=502, detail=f"chain failed during profiling: {exc}"
            ) from exc
        finally:
            runner.close()

        profiles = []
        for seq_len in request.seq_lens:
            subset = [r for r in profiler.records if r.seq_len == seq_len and r.phase == "prefill"]
            summaries = summarize_by_hop(subset)
            profiles.append(
                SeqLenProfile(
                    seq_len=seq_len,
                    verdict=bandwidth_verdict(summaries),
                    hops=[HopSummaryMetrics.from_summary(s) for s in summaries],
                )
            )

        return ProfileResponse(
            model_id=config.model_id,
            codec=config.codec,
            repeats=request.repeats,
            profiles=profiles,
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        """Liveness of the API process itself, not of the chain.

        Kept separate from /topology on purpose: this answering while /topology
        returns 503 is the signal that the API is fine and the shard nodes are
        not, which is the single most common failure while developing.
        """
        return {"status": "ok"}

    return app
