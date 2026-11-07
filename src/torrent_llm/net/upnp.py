"""Asking the home router to forward a port (UPnP IGD), with the standard library only.

Most home routers will open a port on request. When it works, a laptop behind the
router becomes directly reachable from anywhere, and nobody has to relay for it.
When it does not (phone hotspots, carrier-grade NAT, routers with UPnP off) nothing
is lost: the device falls back to relays. So every function here is best effort:
it returns ``None``/``False`` instead of raising, and never takes long.

The protocol, in three steps:

1. **SSDP**: send an ``M-SEARCH`` datagram to 239.255.255.250:1900 asking for
   ``urn:schemas-upnp-org:device:InternetGatewayDevice:1``; a router answers with a
   ``LOCATION:`` header pointing at its device description (an XML file over HTTP).
2. **Description**: fetch that XML and find the ``controlURL`` of a service whose
   ``serviceType`` is ``urn:schemas-upnp-org:service:WANIPConnection:1`` or ``:2``
   or ``WANPPPConnection:1`` (resolve it against ``URLBase`` or the location).
3. **SOAP**: POST ``GetExternalIPAddress`` and ``AddPortMapping`` /
   ``DeletePortMapping`` to the control URL.

Everything a router sends is untrusted: responses are capped at 64 KiB, XML with a
``<!DOCTYPE`` or ``<!ENTITY`` is refused (entity-expansion attacks), and only
``http://`` URLs on the same host as the SSDP responder are followed (a malicious
device must not turn this into a request to anywhere).
"""

from __future__ import annotations

from dataclasses import dataclass

SSDP_ADDRESS = ("239.255.255.250", 1900)
MAX_RESPONSE_BYTES = 64 * 1024
SERVICE_TYPES = (
    "urn:schemas-upnp-org:service:WANIPConnection:2",
    "urn:schemas-upnp-org:service:WANIPConnection:1",
    "urn:schemas-upnp-org:service:WANPPPConnection:1",
)


@dataclass(frozen=True)
class Gateway:
    control_url: str
    service_type: str
    #: This machine's address on the router's network (the mapping's target).
    local_ip: str


@dataclass(frozen=True)
class Mapping:
    external_ip: str
    external_port: int
    internal_port: int
    gateway: Gateway

    @property
    def address(self) -> str:
        return f"tcp://{self.external_ip}:{self.external_port}"


async def discover_gateway(
    *, timeout: float = 2.0, ssdp_address: tuple[str, int] = SSDP_ADDRESS
) -> Gateway | None:
    """Find the router's port-mapping service, or ``None`` within ``timeout`` seconds.

    ``ssdp_address`` is overridable so tests can answer from 127.0.0.1 instead of multicast.
    """
    raise NotImplementedError


async def external_ip(gateway: Gateway, *, timeout: float = 2.0) -> str | None:
    """The router's public IPv4 address. ``None`` if unknown or if it is itself private
    (a router behind another NAT cannot make us reachable)."""
    raise NotImplementedError


async def add_port_mapping(
    gateway: Gateway,
    *,
    external_port: int,
    internal_port: int,
    lease_s: int = 3600,
    description: str = "Torrent-LLM",
    timeout: float = 2.0,
) -> bool:
    raise NotImplementedError


async def delete_port_mapping(
    gateway: Gateway, *, external_port: int, timeout: float = 2.0
) -> bool:
    raise NotImplementedError


async def map_port(
    internal_port: int,
    *,
    lease_s: int = 3600,
    timeout: float = 3.0,
    ssdp_address: tuple[str, int] = SSDP_ADDRESS,
) -> Mapping | None:
    """Discover the router, map ``internal_port`` to the same external port (or, if the
    router refuses, to ``internal_port + 1 .. + 9``), and return the mapping with the
    public address. ``None`` at the first step that fails. Never raises."""
    raise NotImplementedError
