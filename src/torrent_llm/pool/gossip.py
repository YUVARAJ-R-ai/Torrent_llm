"""Swapping membership tables with other devices over plain HTTP.

Each round a device sends its whole table to a couple of others and gets theirs
back, then merges what it learned. The table is a few hundred bytes per device,
so sending all of it is cheaper than being clever about what changed.

Only the standard library is used, so a device needs nothing beyond what it
already has to join a pool.
"""

from __future__ import annotations

import json
import logging
import random
import urllib.error
import urllib.request
from collections.abc import Sequence

from torrent_llm.pool.members import LEFT, Member, MemberTable

logger = logging.getLogger(__name__)

TOKEN_HEADER = "x-pool-token"


def exchange(
    url: str,
    sender: str,
    digest: list[dict],
    *,
    token: str | None = None,
    timeout: float = 0.8,
) -> list[Member] | None:
    """Send ``digest`` to the device at ``url`` and return its table, or ``None``.

    ``None`` means it did not answer in time, which is information the caller
    uses (a device that never answers stops being asked about). It is not an error.
    """
    body = json.dumps({"sender": sender, "members": digest}).encode()
    request = urllib.request.Request(
        f"{url}/gossip",
        data=body,
        headers={"content-type": "application/json", **({TOKEN_HEADER: token} if token else {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            payload = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None
    return [Member.from_dict(m) for m in payload.get("members", [])]


def choose_targets(
    table: MemberTable,
    seeds: Sequence[str],
    *,
    fanout: int = 2,
    rng: random.Random | None = None,
) -> list[str]:
    """Which devices to talk to this round, as control URLs.

    Mostly devices believed alive. One suspected-down device is added when there
    is one, so a partition heals and a restarted device is noticed. The seeds are
    used when the table knows no one yet, which is how a new device finds its
    first neighbour.
    """
    rng = rng or random
    others = [m for m in table.members() if m.name != table.me and m.state != LEFT]
    alive = [m for m in others if table.liveness(m.name) == "alive"]
    gone = [m for m in others if table.liveness(m.name) != "alive"]

    urls = [m.control_url for m in rng.sample(alive, min(fanout, len(alive)))]
    if gone:
        urls.append(rng.choice(gone).control_url)
    if not alive:
        urls.extend(f"http://{s}" for s in seeds)
    return list(dict.fromkeys(urls))
