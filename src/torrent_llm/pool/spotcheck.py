"""Catching a device that returns wrong answers, by asking another copy now and then.

A pass through the pool picks one token. To check it, the same input (the whole text
so far, no cache) is run through the same route **with exactly one stage swapped**
for another copy of that stage. Only the swapped stage differs, so if the tokens
disagree, one of those two copies is lying (or broken). A third copy of that stage,
or this device itself if it holds the stage, breaks the tie, and the one that
disagrees with the majority gets a failed check in the trust store. Two failed checks
ban it on this device (``trust.py``).

Checks run on a background thread, one at a time, so a person waiting for an answer
never waits for a check. A check that cannot be done (no second copy of any stage
this device does not hold itself) is simply skipped: unverifiable, not guilty.

The comparison is on the chosen token, under the same deterministic sampling the
pass used. Different hardware rounds differently, so on a near-tie two honest copies
can very rarely pick different tokens; arbitration plus two strikes keeps that from
banning an honest device in practice.
"""

from __future__ import annotations

import logging
import random
import threading
from collections.abc import Callable

import torch

from torrent_llm.pool.router import Replica, replicas_of
from torrent_llm.pool.trust import TrustStore, Verifier
from torrent_llm.runner import ChainRunner
from torrent_llm.sampling import Sampling

logger = logging.getLogger(__name__)

#: Builds a chain over a route (the runner's own ``_chain``).
ChainFactory = Callable[[list[Replica]], ChainRunner]


class SpotChecker:
    def __init__(
        self,
        trust: TrustStore,
        verifier: Verifier,
        *,
        me: str | None,
        members: Callable[[], list],
        chain: ChainFactory,
        rng: random.Random | None = None,
    ) -> None:
        self.trust = trust
        self.verifier = verifier
        self.me = me
        self._members = members
        self._chain = chain
        self._rng = rng or random.Random()
        self._busy = threading.Lock()
        self.checks_run = 0

    # --- choosing ------------------------------------------------------------

    def plan(self, max_new_tokens: int) -> int | None:
        """The pass index of a request to check, or ``None`` to leave it alone."""
        if not self.verifier.check_this_request():
            return None
        return self.verifier.pick_pass(max_new_tokens)

    def _alternatives(self, replica: Replica, exclude: set[str]) -> list[Replica]:
        return [
            r
            for r in replicas_of(replica.stage, self._members(), exclude=exclude)
            if r.member != replica.member
        ]

    # --- checking --------------------------------------------------------------

    def submit(
        self, route: list[Replica], ids: torch.Tensor, token_id: int, sampling: Sampling
    ) -> bool:
        """Check this pass in the background. Returns False if a check is already running
        or nothing on the route can be checked."""
        candidates = [r for r in route if r.member != self.me and self._alternatives(r, set())]
        if not candidates or not self._busy.acquire(blocking=False):
            return False
        suspect = self._rng.choice(candidates)
        thread = threading.Thread(
            target=self._run,
            args=(route, suspect, ids.detach().clone(), token_id, sampling),
            name="spot-check",
            daemon=True,
        )
        thread.start()
        return True

    def _run(
        self,
        route: list[Replica],
        suspect: Replica,
        ids: torch.Tensor,
        token_id: int,
        sampling: Sampling,
    ) -> None:
        try:
            self.checks_run += 1
            alternatives = self._alternatives(suspect, {suspect.member})
            if not alternatives:
                return
            other = self._rng.choice(alternatives)
            second = self._token(self._swap(route, suspect, other), ids, sampling)
            if second is None:
                return
            if second == token_id:
                self.trust.record_check(suspect.member, True)
                self.trust.record_check(other.member, True)
                return
            # They disagree: one of the two is wrong. Ask a third copy to decide.
            arbiters = self._alternatives(suspect, {suspect.member, other.member})
            if not arbiters:
                logger.info(
                    "spot-check: %s and %s disagree on stage %d and nobody else can decide",
                    suspect.member[:8],
                    other.member[:8],
                    suspect.stage,
                )
                return
            third = self._token(
                self._swap(route, suspect, self._rng.choice(arbiters)), ids, sampling
            )
            if third is None:
                return
            if third not in (token_id, second):
                logger.info("spot-check: three copies gave three answers; nobody is blamed")
                return
            # The arbiter sides with one of the two; the other one is the odd one out.
            liar, honest = (suspect, other) if third == second else (other, suspect)
            logger.warning(
                "spot-check: %s returned a different result from two other copies of stage %d",
                liar.member[:8],
                liar.stage,
            )
            self.trust.record_check(liar.member, False)
            self.trust.record_check(honest.member, True)
        except Exception as exc:  # noqa: BLE001 - a failed check must never hurt the pool
            logger.debug("spot-check could not finish: %s", exc)
        finally:
            self._busy.release()

    @staticmethod
    def _swap(route: list[Replica], old: Replica, new: Replica) -> list[Replica]:
        return [new if r is old else r for r in route]

    def _token(self, route: list[Replica], ids: torch.Tensor, sampling: Sampling) -> int | None:
        """The token this route picks for ``ids``, statelessly (no cache), or ``None``."""
        chain = self._chain(route)
        try:
            result = chain.forward(ids, use_cache=False, logits_keep_last=1, sampling=sampling)
        except Exception as exc:  # noqa: BLE001 - an unreachable copy proves nothing
            logger.debug("spot-check route failed: %s", exc)
            return None
        finally:
            chain.close()
        if result.token_ids is None:
            return None
        return int(result.token_ids.reshape(-1)[0].item())
