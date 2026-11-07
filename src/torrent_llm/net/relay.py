"""Reaching a device nobody can connect to, through one that everybody can.

A laptop on a phone hotspot or behind carrier-grade NAT can make outgoing
connections but accepts none. It keeps one connection open to a reachable peer
(a *relay*) and advertises ``relay://<relay peer id>``. Someone who wants to reach
it connects to the relay and asks for a *circuit*; the relay pipes bytes between
the two connections. The two ends then run the normal secure handshake *through*
the circuit, so the relay forwards ciphertext it cannot read or alter undetected.

Wire (all over existing secure connections to the relay):

=====================  ==========================================================
``relay.reserve``      unreachable peer -> relay: ``{} -> {"ttl": seconds}``.
                       Keep my connection; route circuits for me. Renewed every
                       ``ttl / 2``. Valid while that connection lives.
``relay.connect``      dialer -> relay: ``{"target": peer id} -> {"circuit": id}``.
``relay.incoming``     relay -> target: ``{"from": peer id, "circuit": id} -> {}``.
                       The target answers once its end of the circuit exists.
CIRCUIT_DATA (16)      extension frame, ``id`` = circuit id on this connection,
                       body = bytes of the end-to-end stream.
CIRCUIT_CLOSE (17)     extension frame, header ``{"reason": str}``.
CIRCUIT_ACK (18)       extension frame, header ``{"n": bytes consumed}`` (flow control).
=====================  ==========================================================

Circuit ids are chosen by the relay, separately on each of its two connections.
Flow control: a sender may have at most ``WINDOW_BYTES`` unacknowledged bytes in
flight on a circuit; the receiver acknowledges as its reader consumes, and the relay
forwards acknowledgements like data. So a relay never buffers more than one window
per direction per circuit, and a slow reader slows the writer instead of filling
anyone's memory.

Limits a relay enforces (``RelayLimits``): reservations, circuits in total and per
peer, bytes per circuit, idle time per circuit. A refused request gets
``RpcError("busy")`` (limits) or ``RpcError("not_found")`` (no reservation for target)
or ``RpcError("denied")`` (relaying switched off).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from torrent_llm.net.node import Node
from torrent_llm.net.records import Address
from torrent_llm.net.stream import ByteStream

CIRCUIT_DATA, CIRCUIT_CLOSE, CIRCUIT_ACK = 16, 17, 18
WINDOW_BYTES = 4 * 1024 * 1024
#: Bytes per CIRCUIT_DATA frame.
CHUNK_BYTES = 256 * 1024
RESERVATION_TTL_S = 600


@dataclass(frozen=True)
class RelayLimits:
    max_reservations: int = 128
    max_circuits: int = 256
    max_circuits_per_peer: int = 8
    max_bytes_per_circuit: int = 8 * 1024 * 1024 * 1024
    idle_timeout_s: float = 300.0


class CircuitStream:
    """One end of a circuit, as a :class:`~torrent_llm.net.stream.ByteStream`.

    ``read_exactly`` waits for data (reporting progress per chunk received), sends
    CIRCUIT_ACK as it consumes, and raises ``StreamClosed`` once the circuit is closed
    and drained. ``write`` splits into CHUNK_BYTES frames and waits while the window
    is full. ``close`` sends CIRCUIT_CLOSE once; closing the underlying connection
    closes every circuit on it.
    """

    @property
    def remote(self) -> str:
        raise NotImplementedError

    async def read_exactly(self, n: int, *, on_progress: Callable[[], None] | None = None) -> bytes:
        raise NotImplementedError

    async def write(self, data: bytes) -> None:
        raise NotImplementedError

    async def close(self) -> None:
        raise NotImplementedError


class Relay:
    """Relaying for one node: the service (if ``serve``), the client side (reserving on
    relays and accepting circuits through them), and the dialer for ``relay://``.

    ``attach()`` registers ``relay.reserve``/``relay.connect`` (only when ``serve``) and
    ``relay.incoming``, installs the node's extension handler, and adds the dialer.
    ``serve`` also sets the node's record attr ``"relay": True``.

    ``reserve_on(relay_peer_ids)`` reserves on each (skipping failures), keeps the
    reservations renewed and re-made after a relay connection drops (trying the next
    candidate given to ``set_candidates``), and keeps
    ``node.set_relay_addresses([f"relay://{id}" ...])`` equal to the live reservations.
    """

    def __init__(
        self, node: Node, *, serve: bool = False, limits: RelayLimits | None = None
    ) -> None:
        raise NotImplementedError

    def attach(self) -> None:
        raise NotImplementedError

    def detach(self) -> None:
        """Stop renewing, close every circuit, cancel tasks. Idempotent."""
        raise NotImplementedError

    @property
    def serving(self) -> bool:
        raise NotImplementedError

    def set_serving(self, serve: bool) -> None:
        raise NotImplementedError

    def reserve_on(self, relay_peer_ids: list[str], *, want: int = 2) -> list[str]:
        """Synchronous; returns the relays now holding a reservation (at most ``want``)."""
        raise NotImplementedError

    def set_candidates(self, relay_peer_ids: list[str]) -> None:
        raise NotImplementedError

    def reservations(self) -> list[str]:
        raise NotImplementedError

    def stats(self) -> dict[str, int]:
        """``{"reservations", "circuits", "bytes_relayed"}`` for the service side."""
        raise NotImplementedError

    async def dial(
        self, node: Node, target: str, address: Address, timeout: float
    ) -> ByteStream | None:
        """The node dialer: ``None`` for non-relay addresses."""
        raise NotImplementedError
