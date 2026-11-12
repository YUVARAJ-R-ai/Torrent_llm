"""Is this device reachable from outside? Ask peers to try connecting back.

A device cannot tell from the inside whether a router forwards its port. So it asks
peers that already know it to open a *fresh* connection to the addresses it thinks
might work, and to report which succeeded. A confirmed address goes into the
device's record; with none, the device is ``private`` and relies on relays.

``autonat.dialback``: ``{"addrs": ["tcp://ip:port", ...]} -> {"reachable": [...]}``.
The helper dials back **only addresses whose IP equals the caller's own observed
IP** (otherwise a stranger could make it connect anywhere: an amplification and
port-scanning vector), at most 4 of them, each with a short timeout, using a new
connection that runs the full handshake with ``expect_peer`` = the caller and is
closed straight after (never adopted into the node).

Candidate addresses: each IP peers report observing (``node.observed_addresses``)
combined with this node's listen port, plus any address given explicitly (a UPnP
mapping). Private-range candidates are skipped: they prove nothing about the internet.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging

from torrent_llm.net.channel import initiate
from torrent_llm.net.node import Node
from torrent_llm.net.records import TcpAddress, parse_address
from torrent_llm.net.rpc import CallContext, RpcError
from torrent_llm.net.stream import TcpStream

logger = logging.getLogger(__name__)

DIALBACK_TIMEOUT_S = 3.0
MAX_DIALBACK_ADDRS = 4
MAX_CONCURRENT_DIALS = 16

PUBLIC, PRIVATE, UNKNOWN = "public", "private", "unknown"

#: When True, candidates() keeps private/loopback IPs so loopback tests can exercise the full path.
allow_private_candidates: bool = False


class AutoNat:
    """Reachability for one node.

    ``attach()`` registers ``autonat.dialback`` (when ``serve``).
    ``check(helpers, extra=())`` asks each helper peer (up to 3) to dial back the
    candidates; returns the confirmed addresses, updates ``status`` (``public`` if any
    address was confirmed by at least one helper, ``private`` if every helper answered
    and none succeeded, ``unknown`` if no helper answered) and, when public, sets them via
    ``node.set_advertised`` (keeping any extra advertised addresses) and the record attr
    ``"public": True``.
    """

    def __init__(self, node: Node, *, serve: bool = True) -> None:
        self._node = node
        self._serve = serve
        self._status: str = UNKNOWN
        self._reachable: list[str] = []
        self._attached: bool = False
        self._dial_sem: asyncio.Semaphore | None = None
        self._inflight_ips: set[str] = set()
        self._added: list[str] = []

    @property
    def _dial_semaphore(self) -> asyncio.Semaphore:
        """Lazily initialize and return the concurrency limit semaphore for dials."""
        if self._dial_sem is None:
            self._dial_sem = asyncio.Semaphore(MAX_CONCURRENT_DIALS)
        return self._dial_sem

    def attach(self) -> None:
        """Register the autonat.dialback RPC handler if serve is True."""
        if self._serve and not self._attached:
            self._node.register("autonat.dialback", self._handle_dialback, blocking=False)
            self._attached = True

    @property
    def status(self) -> str:
        """Reachability status: 'public', 'private', or 'unknown'."""
        return self._status

    @property
    def reachable(self) -> list[str]:
        """Confirmed reachable addresses from the last check."""
        return list(self._reachable)

    def candidates(self, extra: tuple[str, ...] = ()) -> list[str]:
        """Build candidate dialback addresses from observed addresses and explicit extras.

        Combines observed IPs with this node's listen port, appends extras, drops private
        ranges unless allow_private_candidates is True, and deduplicates while preserving order.
        """
        raw: list[str] = []
        if self._node.listen_port > 0:
            for obs in self._node.observed_addresses():
                ip = obs.rsplit(":", 1)[0].strip("[]")
                if not ip:
                    continue
                cand = (
                    f"tcp://[{ip}]:{self._node.listen_port}"
                    if ":" in ip
                    else f"tcp://{ip}:{self._node.listen_port}"
                )
                raw.append(cand)
        if extra:
            raw.extend(extra)

        seen: set[str] = set()
        result: list[str] = []
        for item in raw:
            if not isinstance(item, str):
                continue
            try:
                parsed = parse_address(item)
            except ValueError:
                continue
            if not isinstance(parsed, TcpAddress):
                continue
            if parsed.is_private and not allow_private_candidates:
                continue
            canon = str(parsed)
            if canon not in seen:
                seen.add(canon)
                result.append(canon)
        return result

    async def _handle_dialback(
        self, ctx: CallContext, header: dict, body: bytes
    ) -> tuple[dict, bytes]:
        """Dial back caller's candidates whose IP matches the caller's observed IP."""
        # A refusal is an error, never {"reachable": []}: an empty list means "I dialled
        # and failed", which would wrongly make the caller think it is private.
        raw_addrs = header.get("addrs")
        if not isinstance(raw_addrs, list):
            raise RpcError("bad_request", "addrs must be a list")

        remote = ctx.remote
        if not remote or remote == "self":
            raise RpcError("denied", "no remote address to check against")

        # Do not dial back if the caller's connection arrived over a relay
        if any(
            c.via_relay is not None
            for c in self._node.connections()
            if c.peer_id == ctx.peer_id and c.remote == remote
        ):
            logger.debug("refusing dialback for relayed connection from %s", ctx.peer_id)
            raise RpcError("denied", "cannot check a device that came in through a relay")

        caller_ip_str = remote.rsplit(":", 1)[0].strip("[]")
        try:
            caller_ip = ipaddress.ip_address(caller_ip_str)
        except ValueError:
            logger.debug("caller remote %r is not a valid IP", remote)
            raise RpcError("denied", "caller address is not an IP") from None

        # Prevent concurrent dialbacks from the same caller IP
        canonical_ip_str = str(caller_ip)
        if canonical_ip_str in self._inflight_ips:
            logger.debug("dialback already in-flight for %s", canonical_ip_str)
            raise RpcError("busy", "a check for this address is already running")
        self._inflight_ips.add(canonical_ip_str)

        try:
            return await self._execute_dialback(ctx, caller_ip, raw_addrs)
        finally:
            self._inflight_ips.discard(canonical_ip_str)

    async def _execute_dialback(
        self,
        ctx: CallContext,
        caller_ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
        raw_addrs: list,
    ) -> tuple[dict, bytes]:
        """Execute dialbacks to validated candidates matching the caller's IP."""
        valid_addrs: list[TcpAddress] = []
        seen: set[tuple[str, int]] = set()
        for item in raw_addrs:
            if not isinstance(item, str):
                continue
            try:
                parsed = parse_address(item)
            except ValueError:
                continue
            if not isinstance(parsed, TcpAddress):
                continue
            try:
                host_ip = ipaddress.ip_address(parsed.host)
            except ValueError:
                continue
            if host_ip != caller_ip:
                continue
            key = (parsed.host, parsed.port)
            if key in seen:
                continue
            seen.add(key)
            valid_addrs.append(parsed)
            if len(valid_addrs) >= MAX_DIALBACK_ADDRS:
                break

        if not valid_addrs:
            return {"reachable": []}, b""

        async def _try_one(addr: TcpAddress) -> str | None:
            async with self._dial_semaphore:
                try:
                    reader, writer = await asyncio.wait_for(
                        asyncio.open_connection(addr.host, addr.port),
                        timeout=DIALBACK_TIMEOUT_S,
                    )
                except OSError as exc:  # refused, unreachable, timed out
                    logger.debug("dialback connection failed to %s: %s", addr, exc)
                    return None
                except Exception:
                    logger.exception("dialback to %s failed here", addr)
                    return None

                stream = TcpStream(reader, writer)
                ch = None
                try:
                    ch = await initiate(
                        stream,
                        self._node.identity,
                        psk=self._node.config.psk,
                        expect_peer=ctx.peer_id,
                        min_pow_bits=self._node.config.min_pow_bits,
                        timeout=DIALBACK_TIMEOUT_S,
                    )
                    return str(addr)
                except OSError as exc:  # handshake and stream errors are ConnectionErrors
                    logger.debug(
                        "dialback handshake failed with %s at %s: %s", ctx.peer_id, addr, exc
                    )
                    return None
                except Exception:
                    logger.exception("dialback handshake to %s failed here", addr)
                    return None
                finally:
                    if ch is not None:
                        await ch.close()
                    else:
                        await stream.close()

        results = await asyncio.gather(*(_try_one(a) for a in valid_addrs))
        reachable = [r for r in results if isinstance(r, str)]
        return {"reachable": reachable}, b""

    async def acheck(self, helpers: list[str], extra: tuple[str, ...] = ()) -> list[str]:
        """Asynchronously query helpers (up to 3) to dial back our candidate addresses."""
        cands = self.candidates(extra)
        cands_set = set(cands)

        selected_helpers: list[str] = []
        for h in helpers:
            if h != self._node.peer_id and h not in selected_helpers:
                selected_helpers.append(h)
            if len(selected_helpers) == 3:
                break

        if not selected_helpers or not cands:  # nothing to ask, or nothing to test
            self._status = UNKNOWN
            self._reachable = []
            return []

        async def _query_helper(helper: str) -> list[str] | None:
            try:
                res_header, _ = await self._node.acall(
                    helper,
                    "autonat.dialback",
                    {"addrs": cands},
                    timeout=DIALBACK_TIMEOUT_S * 2 + 1,
                )
                if isinstance(res_header, dict):
                    reachable_raw = res_header.get("reachable")
                    if isinstance(reachable_raw, list):
                        return [a for a in reachable_raw if isinstance(a, str) and a in cands_set]
                return []
            except RpcError as exc:  # unreachable, busy or refusing: not an answer
                logger.debug("querying helper %s failed: %s", helper, exc)
                return None
            except Exception:
                logger.exception("querying helper %s failed here", helper)
                return None

        results = await asyncio.gather(*(_query_helper(h) for h in selected_helpers))

        answered_count = 0
        confirmed_set: set[str] = set()
        for r in results:
            if r is not None:
                answered_count += 1
                confirmed_set.update(r)

        confirmed_list = [c for c in cands if c in confirmed_set]

        if confirmed_list:
            self._status = PUBLIC
            self._reachable = confirmed_list
            existing = list(getattr(self._node, "_advertised", []))
            base_advertised = [a for a in existing if a not in self._added]
            new_advertised = list(base_advertised)
            for addr in confirmed_list:
                if addr not in new_advertised:
                    new_advertised.append(addr)
            # Only what this check put there is ours to take away later.
            self._added = [a for a in confirmed_list if a not in base_advertised]
            self._node.set_advertised(new_advertised)
            self._node.set_attr("public", True)
        elif answered_count == len(selected_helpers):
            self._status = PRIVATE
            self._reachable = []
            if self._added:
                existing = list(getattr(self._node, "_advertised", []))
                clean_advertised = [a for a in existing if a not in self._added]
                self._added.clear()
                self._node.set_advertised(clean_advertised)
            self._node.set_attr("public", False)
        else:
            self._status = UNKNOWN
            self._reachable = []

        return list(self._reachable)

    def check(self, helpers: list[str], extra: tuple[str, ...] = ()) -> list[str]:
        """Synchronously check reachability via helpers."""
        timeout = (DIALBACK_TIMEOUT_S * 2 + 1) + 5.0
        return self._node.run(self.acheck(helpers, extra=extra), timeout=timeout)
