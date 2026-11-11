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

import asyncio
import ipaddress
import logging
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import xml.sax.saxutils
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

SSDP_ADDRESS = ("239.255.255.250", 1900)
MAX_RESPONSE_BYTES = 64 * 1024
SERVICE_TYPES = (
    "urn:schemas-upnp-org:service:WANIPConnection:2",
    "urn:schemas-upnp-org:service:WANIPConnection:1",
    "urn:schemas-upnp-org:service:WANPPPConnection:1",
)

_M_SEARCH_PAYLOAD = (
    b"M-SEARCH * HTTP/1.1\r\n"
    b"HOST: 239.255.255.250:1900\r\n"
    b'MAN: "ssdp:discover"\r\n'
    b"MX: 2\r\n"
    b"ST: urn:schemas-upnp-org:device:InternetGatewayDevice:1\r\n"
    b"\r\n"
)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse HTTP redirects to prevent reaching arbitrary hosts."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


_OPENER = urllib.request.build_opener(
    urllib.request.ProxyHandler({}),
    _NoRedirect(),
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


def _local_tag(tag: str) -> str:
    """Strip XML namespace prefix if present."""
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def _local_ip_for(router_host: str) -> str | None:
    """The local IPv4 address the OS uses to route to router_host, or None."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect((router_host, 9))
        return probe.getsockname()[0]
    except OSError:
        return None
    finally:
        probe.close()


def _safe_xml(raw: bytes) -> ET.Element | None:
    """Parse XML bytes safely, refusing UTF-16/32 encodings, DOCTYPE, and ENTITY."""
    if not raw:
        return None
    if b"\x00" in raw or raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return None
    raw_upper = raw.upper()
    if b"<!DOCTYPE" in raw_upper or b"<!ENTITY" in raw_upper:
        return None
    try:
        return ET.fromstring(raw)
    except Exception:
        return None


def _read_capped(resp: Any, deadline: float) -> bytes | None:
    """Read up to MAX_RESPONSE_BYTES + 1 before deadline.

    Returns None if oversize or timed out.
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        if time.monotonic() > deadline:
            return None
        chunk = resp.read(min(4096, MAX_RESPONSE_BYTES + 1 - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > MAX_RESPONSE_BYTES:
            return None
    return b"".join(chunks)


def _http_get(url: str, deadline: float) -> bytes | None:
    """Fetch an HTTP URL synchronously, reading at most MAX_RESPONSE_BYTES + 1 before deadline."""
    rem = deadline - time.monotonic()
    if rem <= 0:
        return None
    req = urllib.request.Request(url, headers={"User-Agent": "Torrent-LLM"})
    try:
        with _OPENER.open(req, timeout=rem) as resp:
            cl = resp.headers.get("Content-Length")
            if cl is not None and cl.isdigit() and int(cl) > MAX_RESPONSE_BYTES:
                return None
            return _read_capped(resp, deadline)
    except urllib.error.HTTPError as err:
        logger.debug("HTTP GET %s failed with code %s", url, err.code)
        return None
    except Exception as exc:
        logger.debug("HTTP GET %s failed: %s", url, exc)
        return None


def _soap_call(
    control_url: str,
    service_type: str,
    action: str,
    args: list[tuple[str, str]],
    deadline: float,
) -> tuple[int, bytes]:
    """Execute a SOAP request, returning (http_status, response_body)."""
    rem = deadline - time.monotonic()
    if rem <= 0:
        return 0, b""
    args_xml = "".join(f"<{k}>{xml.sax.saxutils.escape(str(v))}</{k}>" for k, v in args)
    body = (
        '<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/" '
        's:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/">'
        f'<s:Body><u:{action} xmlns:u="{service_type}">{args_xml}</u:{action}></s:Body>'
        "</s:Envelope>"
    ).encode()

    headers = {
        "Content-Type": 'text/xml; charset="utf-8"',
        "SOAPAction": f'"{service_type}#{action}"',
        "Content-Length": str(len(body)),
        "User-Agent": "Torrent-LLM",
    }
    req = urllib.request.Request(control_url, data=body, headers=headers, method="POST")
    try:
        with _OPENER.open(req, timeout=rem) as resp:
            data = _read_capped(resp, deadline)
            if data is None:
                return 0, b""
            status = getattr(resp, "status", resp.getcode())
            return status, data
    except urllib.error.HTTPError as err:
        if 300 <= err.code < 400:
            return 0, b""
        return err.code, b""
    except Exception as exc:
        logger.debug("SOAP call %s failed: %s", action, exc)
        return 0, b""


class _SSDPClientProtocol(asyncio.DatagramProtocol):
    """Collect incoming SSDP UDP datagrams into a bounded asyncio queue."""

    def __init__(self, maxsize: int = 32) -> None:
        self.queue: asyncio.Queue[tuple[bytes, tuple[str, int]]] = asyncio.Queue(maxsize=maxsize)

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        try:
            self.queue.put_nowait((data, addr))
        except asyncio.QueueFull:
            pass


async def _inspect_ssdp_reply(
    data: bytes, addr: tuple[str, int], deadline: float
) -> Gateway | None:
    """Inspect an SSDP datagram and fetch the description XML if valid."""
    text = data.decode("latin-1", errors="replace")
    lines = text.splitlines()
    if not lines:
        return None

    parts = lines[0].strip().split()
    if len(parts) < 2 or parts[0].upper() != "HTTP/1.1" or parts[1] != "200":
        return None

    location: str | None = None
    for line in lines[1:]:
        line = line.strip()
        if not line:
            break
        if ":" in line:
            k, v = line.split(":", 1)
            if k.strip().lower() == "location":
                location = v.strip()
                break

    if not location:
        return None

    parsed_loc = urllib.parse.urlsplit(location)
    if parsed_loc.scheme.lower() != "http":
        return None

    reply_ip = addr[0]
    if (parsed_loc.hostname or "").lower() != reply_ip.lower():
        return None

    rem = deadline - time.monotonic()
    if rem <= 0:
        return None

    loop = asyncio.get_running_loop()
    try:
        raw_xml = await asyncio.wait_for(
            loop.run_in_executor(None, _http_get, location, deadline),
            timeout=rem,
        )
    except Exception as exc:
        logger.debug("Fetching description from %s failed: %s", location, exc)
        return None

    if raw_xml is None:
        return None

    root = _safe_xml(raw_xml)
    if root is None:
        return None

    url_base: str | None = None
    for elem in root.iter():
        if _local_tag(elem.tag) == "URLBase" and elem.text:
            text_val = elem.text.strip()
            if text_val:
                url_base = text_val
                break

    services: list[tuple[str, str]] = []
    for elem in root.iter():
        if _local_tag(elem.tag) == "service":
            st_text: str | None = None
            ctrl_text: str | None = None
            for child in elem:
                t = _local_tag(child.tag)
                if t == "serviceType" and child.text:
                    st_text = child.text.strip()
                elif t == "controlURL" and child.text:
                    ctrl_text = child.text.strip()
            if st_text and ctrl_text:
                services.append((st_text, ctrl_text))

    local_ip = _local_ip_for(reply_ip)
    if local_ip is None:
        return None

    base = urllib.parse.urljoin(location, url_base) if url_base else location
    for target_st in SERVICE_TYPES:
        for st, ctrl in services:
            if st == target_st:
                control_url = urllib.parse.urljoin(base, ctrl)
                p_ctrl = urllib.parse.urlsplit(control_url)
                if (
                    p_ctrl.scheme.lower() == "http"
                    and (p_ctrl.hostname or "").lower() == (parsed_loc.hostname or "").lower()
                ):
                    return Gateway(
                        control_url=control_url,
                        service_type=target_st,
                        local_ip=local_ip,
                    )
    return None


async def discover_gateway(
    *, timeout: float = 2.0, ssdp_address: tuple[str, int] = SSDP_ADDRESS
) -> Gateway | None:
    """Find the router's port-mapping service, or ``None`` within ``timeout`` seconds.

    ``ssdp_address`` is overridable so tests can answer from 127.0.0.1 instead of multicast.
    """
    loop = asyncio.get_running_loop()
    protocol = _SSDPClientProtocol(maxsize=32)
    try:
        transport, _ = await loop.create_datagram_endpoint(
            lambda: protocol, local_addr=("0.0.0.0", 0)
        )
    except OSError as exc:
        logger.debug("Failed to create SSDP datagram endpoint: %s", exc)
        return None

    deadline = time.monotonic() + timeout
    try:
        transport.sendto(_M_SEARCH_PAYLOAD, ssdp_address)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                data, addr = await asyncio.wait_for(protocol.queue.get(), timeout=remaining)
            except TimeoutError:
                return None

            fetch_deadline = time.monotonic() + remaining
            try:
                gateway = await _inspect_ssdp_reply(data, addr, fetch_deadline)
            except Exception as exc:
                logger.debug("Error processing SSDP reply from %s: %s", addr, exc)
                gateway = None

            if gateway is not None:
                return gateway
    except Exception as exc:
        logger.debug("SSDP discovery error: %s", exc)
        return None
    finally:
        transport.close()


async def _soap(
    gateway: Gateway, action: str, args: list[tuple[str, str]], timeout: float
) -> tuple[int, bytes]:
    """Run one SOAP action on the router off the event loop, bounded by ``timeout``."""
    loop = asyncio.get_running_loop()
    deadline = time.monotonic() + timeout
    return await asyncio.wait_for(
        loop.run_in_executor(
            None,
            _soap_call,
            gateway.control_url,
            gateway.service_type,
            action,
            args,
            deadline,
        ),
        timeout=timeout,
    )


async def external_ip(gateway: Gateway, *, timeout: float = 2.0) -> str | None:
    """The router's public IPv4 address. ``None`` if unknown or if it is itself private
    (a router behind another NAT cannot make us reachable)."""
    try:
        status, data = await _soap(gateway, "GetExternalIPAddress", [], timeout)
        if status != 200 or not data:
            return None

        root = _safe_xml(data)
        if root is None:
            return None

        ip_text: str | None = None
        for elem in root.iter():
            if _local_tag(elem.tag) == "NewExternalIPAddress" and elem.text:
                ip_text = elem.text.strip()
                break

        if not ip_text:
            return None

        ip_obj = ipaddress.ip_address(ip_text)
        if isinstance(ip_obj, ipaddress.IPv4Address) and ip_obj.is_global:
            return str(ip_obj)
        return None
    except Exception as exc:
        logger.debug("external_ip error: %s", exc)
        return None


async def add_port_mapping(
    gateway: Gateway,
    *,
    external_port: int,
    internal_port: int,
    lease_s: int = 3600,
    description: str = "Torrent-LLM",
    timeout: float = 2.0,
) -> bool:
    """Ask the router to forward external_port to this machine's internal_port.

    Returns True only on HTTP 200.
    """
    args = [
        ("NewRemoteHost", ""),
        ("NewExternalPort", str(external_port)),
        ("NewProtocol", "TCP"),
        ("NewInternalPort", str(internal_port)),
        ("NewInternalClient", gateway.local_ip),
        ("NewEnabled", "1"),
        ("NewPortMappingDescription", description),
        ("NewLeaseDuration", str(lease_s)),
    ]
    try:
        status, _ = await _soap(gateway, "AddPortMapping", args, timeout)
        return status == 200
    except Exception as exc:
        logger.debug("add_port_mapping error: %s", exc)
        return False


async def delete_port_mapping(
    gateway: Gateway, *, external_port: int, timeout: float = 2.0
) -> bool:
    """Ask the router to remove an external port forwarding mapping; True only on HTTP 200."""
    args = [
        ("NewRemoteHost", ""),
        ("NewExternalPort", str(external_port)),
        ("NewProtocol", "TCP"),
    ]
    try:
        status, _ = await _soap(gateway, "DeletePortMapping", args, timeout)
        return status == 200
    except Exception as exc:
        logger.debug("delete_port_mapping error: %s", exc)
        return False


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
    if not (1 <= internal_port <= 65535):
        return None
    try:
        deadline = time.monotonic() + timeout

        gw = await discover_gateway(timeout=timeout, ssdp_address=ssdp_address)
        if gw is None:
            return None

        mapped_port: int | None = None
        max_port = min(internal_port + 10, 65536)
        for port in range(internal_port, max_port):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            ok = await add_port_mapping(
                gw,
                external_port=port,
                internal_port=internal_port,
                lease_s=lease_s,
                timeout=remaining,
            )
            if ok:
                mapped_port = port
                break

        if mapped_port is None:
            return None

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            await delete_port_mapping(gw, external_port=mapped_port, timeout=1.0)
            return None

        ext_ip = await external_ip(gw, timeout=remaining)
        if ext_ip is None:
            cleanup_timeout = min(max(deadline - time.monotonic(), 0.5), 1.0)
            await delete_port_mapping(gw, external_port=mapped_port, timeout=cleanup_timeout)
            return None

        return Mapping(
            external_ip=ext_ip,
            external_port=mapped_port,
            internal_port=internal_port,
            gateway=gw,
        )
    except Exception as exc:
        logger.debug("map_port failed: %s", exc)
        return None
