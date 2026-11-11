"""Tests for UPnP IGD port mapping."""

from __future__ import annotations

import asyncio
import http.server
import re
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from torrent_llm.net.upnp import (
    MAX_RESPONSE_BYTES,
    Gateway,
    add_port_mapping,
    delete_port_mapping,
    discover_gateway,
    external_ip,
    map_port,
)


def run(coro: Any) -> Any:
    return asyncio.run(coro)


DEFAULT_DESC_XML = (
    b'<?xml version="1.0"?>\n'
    b'<root xmlns="urn:schemas-upnp-org:device-1-0">\n'
    b"  <device>\n"
    b"    <deviceType>urn:schemas-upnp-org:device:InternetGatewayDevice:1</deviceType>\n"
    b"    <serviceList>\n"
    b"      <service>\n"
    b"        <serviceType>urn:schemas-upnp-org:service:WANIPConnection:1</serviceType>\n"
    b"        <controlURL>/control</controlURL>\n"
    b"        <eventSubURL>/event</eventSubURL>\n"
    b"        <SCPDURL>/scpd.xml</SCPDURL>\n"
    b"      </service>\n"
    b"    </serviceList>\n"
    b"  </device>\n"
    b"</root>\n"
)


class FakeRouterState:
    def __init__(self) -> None:
        self.desc_xml: bytes | None = None
        self.external_ip: str = "8.8.4.4"
        self.refuse_ports: set[int] = set()
        self.recorded_posts: list[dict[str, Any]] = []
        self.desc_requests: int = 0
        self.redirect_location: str | None = None
        self.http_port: int = 0
        self.omit_content_length: bool = False
        self.control_delay: float = 0.0


class FakeRouterServer(http.server.ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], RequestHandlerClass: type) -> None:
        super().__init__(server_address, RequestHandlerClass)
        self.state = FakeRouterState()


class FakeRouterHTTPHandler(http.server.BaseHTTPRequestHandler):
    server: FakeRouterServer

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        if self.path == "/desc.xml":
            self.server.state.desc_requests += 1
            if self.server.state.redirect_location is not None:
                self.send_response(302)
                self.send_header("Location", self.server.state.redirect_location)
                self.end_headers()
                return
            content = (
                self.server.state.desc_xml
                if self.server.state.desc_xml is not None
                else DEFAULT_DESC_XML
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/xml")
            if not self.server.state.omit_content_length:
                self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        if self.server.state.control_delay > 0:
            time.sleep(self.server.state.control_delay)

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8", errors="replace")
        self.server.state.recorded_posts.append(
            {
                "path": self.path,
                "headers": self.headers,
                "body": body,
            }
        )

        soap_action = self.headers.get("SOAPAction", "").strip('"')
        action = soap_action.split("#")[-1]

        if action == "GetExternalIPAddress":
            resp_body = (
                f'<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/">'
                f"<s:Body>"
                f"<u:GetExternalIPAddressResponse "
                f'xmlns:u="urn:schemas-upnp-org:service:WANIPConnection:1">'
                f"<NewExternalIPAddress>{self.server.state.external_ip}</NewExternalIPAddress>"
                f"</u:GetExternalIPAddressResponse>"
                f"</s:Body>"
                f"</s:Envelope>"
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", 'text/xml; charset="utf-8"')
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)
            return

        if action == "AddPortMapping":
            m = re.search(r"<NewExternalPort>(\d+)</NewExternalPort>", body)
            port = int(m.group(1)) if m else None
            if port is not None and port in self.server.state.refuse_ports:
                fault = (
                    b'<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/">'
                    b"<s:Body>"
                    b"<s:Fault>"
                    b"<faultcode>s:Client</faultcode>"
                    b"<faultstring>UPnPError</faultstring>"
                    b"<detail>"
                    b'<UPnPError xmlns="urn:schemas-upnp-org:control-1-0">'
                    b"<errorCode>718</errorCode>"
                    b"<errorDescription>ConflictInMappingEntry</errorDescription>"
                    b"</UPnPError>"
                    b"</detail>"
                    b"</s:Fault>"
                    b"</s:Body>"
                    b"</s:Envelope>"
                )
                self.send_response(500)
                self.send_header("Content-Type", 'text/xml; charset="utf-8"')
                self.send_header("Content-Length", str(len(fault)))
                self.end_headers()
                self.wfile.write(fault)
                return

            resp_body = (
                b'<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/">'
                b"<s:Body>"
                b"<u:AddPortMappingResponse "
                b'xmlns:u="urn:schemas-upnp-org:service:WANIPConnection:1"/>'
                b"</s:Body>"
                b"</s:Envelope>"
            )
            self.send_response(200)
            self.send_header("Content-Type", 'text/xml; charset="utf-8"')
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)
            return

        if action == "DeletePortMapping":
            resp_body = (
                b'<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/">'
                b"<s:Body>"
                b"<u:DeletePortMappingResponse "
                b'xmlns:u="urn:schemas-upnp-org:service:WANIPConnection:1"/>'
                b"</s:Body>"
                b"</s:Envelope>"
            )
            self.send_response(200)
            self.send_header("Content-Type", 'text/xml; charset="utf-8"')
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)
            return

        self.send_response(400)
        self.end_headers()


class FakeRouterUDPProtocol(asyncio.DatagramProtocol):
    def __init__(
        self, location: str, *, reply: bool = True, ssdp_status: str = "HTTP/1.1 200 OK"
    ) -> None:
        self.location = location
        self.reply = reply
        self.ssdp_status = ssdp_status
        self.transport: asyncio.DatagramTransport | None = None

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = transport  # type: ignore[assignment]

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        if not self.reply or self.transport is None:
            return
        reply_bytes = (
            f"{self.ssdp_status}\r\n"
            f"CACHE-CONTROL: max-age=1800\r\n"
            f"LOCATION: {self.location}\r\n"
            f"ST: urn:schemas-upnp-org:device:InternetGatewayDevice:1\r\n"
            f"\r\n"
        ).encode("ascii")
        self.transport.sendto(reply_bytes, addr)


@asynccontextmanager
async def fake_router(
    *,
    host: str = "127.0.0.1",
    external_ip: str = "8.8.4.4",
    refuse_ports: set[int] | None = None,
    desc_xml: bytes | None = None,
    custom_location: str | None = None,
    reply_udp: bool = True,
    redirect_location: str | None = None,
    ssdp_status: str = "HTTP/1.1 200 OK",
    omit_content_length: bool = False,
    control_delay: float = 0.0,
) -> AsyncIterator[tuple[int, FakeRouterState]]:
    server = FakeRouterServer((host, 0), FakeRouterHTTPHandler)
    server.state.external_ip = external_ip
    server.state.refuse_ports = refuse_ports or set()
    server.state.desc_xml = desc_xml
    server.state.redirect_location = redirect_location
    server.state.omit_content_length = omit_content_length
    server.state.control_delay = control_delay

    http_port = server.server_address[1]
    server.state.http_port = http_port
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    thread.start()

    location = custom_location or f"http://{host}:{http_port}/desc.xml"
    loop = asyncio.get_running_loop()
    udp_transport, _ = await loop.create_datagram_endpoint(
        lambda: FakeRouterUDPProtocol(location, reply=reply_udp, ssdp_status=ssdp_status),
        local_addr=("127.0.0.1", 0),
    )
    udp_port = udp_transport.get_extra_info("sockname")[1]

    try:
        yield udp_port, server.state
    finally:
        udp_transport.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)


def test_map_port_happy_path() -> None:
    async def _test() -> None:
        async with fake_router() as (udp_port, state):
            mapping = await map_port(5000, ssdp_address=("127.0.0.1", udp_port))
            assert mapping is not None
            assert mapping.external_ip == "8.8.4.4"
            assert mapping.external_port == 5000
            assert mapping.internal_port == 5000
            assert mapping.address == "tcp://8.8.4.4:5000"
            assert mapping.gateway.control_url == f"http://127.0.0.1:{state.http_port}/control"

            add_posts = [
                p
                for p in state.recorded_posts
                if 'AddPortMapping"' in p["headers"].get("SOAPAction", "")
            ]
            assert len(add_posts) == 1
            assert add_posts[0]["path"] == "/control"
            assert add_posts[0]["headers"]["Content-Type"] == 'text/xml; charset="utf-8"'
            assert (
                add_posts[0]["headers"]["SOAPAction"]
                == f'"{mapping.gateway.service_type}#AddPortMapping"'
            )
            body = add_posts[0]["body"]
            assert "<NewExternalPort>5000</NewExternalPort>" in body
            assert "<NewInternalPort>5000</NewInternalPort>" in body
            assert "<NewInternalClient>127.0.0.1</NewInternalClient>" in body
            assert "<NewProtocol>TCP</NewProtocol>" in body
            assert "<NewEnabled>1</NewEnabled>" in body
            assert "<NewPortMappingDescription>Torrent-LLM</NewPortMappingDescription>" in body
            assert "<NewLeaseDuration>3600</NewLeaseDuration>" in body
            assert "<NewRemoteHost></NewRemoteHost>" in body

            ip_posts = [
                p
                for p in state.recorded_posts
                if 'GetExternalIPAddress"' in p["headers"].get("SOAPAction", "")
            ]
            assert len(ip_posts) == 1
            assert ip_posts[0]["path"] == "/control"
            assert ip_posts[0]["headers"]["Content-Type"] == 'text/xml; charset="utf-8"'
            assert (
                ip_posts[0]["headers"]["SOAPAction"]
                == f'"{mapping.gateway.service_type}#GetExternalIPAddress"'
            )

    run(_test())


def test_map_port_router_refusing_first_port() -> None:
    async def _test() -> None:
        async with fake_router(refuse_ports={5000}) as (udp_port, state):
            mapping = await map_port(5000, ssdp_address=("127.0.0.1", udp_port))
            assert mapping is not None
            assert mapping.external_port == 5001
            assert mapping.internal_port == 5000
            assert mapping.address == "tcp://8.8.4.4:5001"

            add_posts = [
                p
                for p in state.recorded_posts
                if 'AddPortMapping"' in p["headers"].get("SOAPAction", "")
            ]
            assert len(add_posts) == 2
            assert "<NewExternalPort>5000</NewExternalPort>" in add_posts[0]["body"]
            assert "<NewExternalPort>5001</NewExternalPort>" in add_posts[1]["body"]

    run(_test())


def test_map_port_private_external_ip() -> None:
    async def _test() -> None:
        async with fake_router(external_ip="10.0.0.1") as (udp_port, state):
            mapping = await map_port(5000, ssdp_address=("127.0.0.1", udp_port))
            assert mapping is None

            delete_posts = [
                p
                for p in state.recorded_posts
                if 'DeletePortMapping"' in p["headers"].get("SOAPAction", "")
            ]
            assert len(delete_posts) == 1
            assert "<NewExternalPort>5000</NewExternalPort>" in delete_posts[0]["body"]

    run(_test())


def test_map_port_invalid_port() -> None:
    async def _test() -> None:
        assert await map_port(0) is None
        assert await map_port(-10) is None
        assert await map_port(65536) is None

    run(_test())


def test_discover_gateway_no_reply_timeout() -> None:
    async def _test() -> None:
        async with fake_router(reply_udp=False) as (udp_port, _):
            t0 = time.monotonic()
            gw = await discover_gateway(timeout=0.1, ssdp_address=("127.0.0.1", udp_port))
            assert gw is None
            elapsed = time.monotonic() - t0
            assert elapsed >= 0.08
            assert elapsed < 0.5

    run(_test())


def test_discover_gateway_refuse_doctype_and_entity_xml() -> None:
    async def _test() -> None:
        # Control assertion: DEFAULT_DESC_XML yields a valid Gateway
        async with fake_router(desc_xml=DEFAULT_DESC_XML) as (udp_port, _):
            control_gw = await discover_gateway(timeout=0.2, ssdp_address=("127.0.0.1", udp_port))
            assert control_gw is not None

        # Uppercase DOCTYPE inserted into otherwise valid desc XML
        doctype_xml = DEFAULT_DESC_XML.replace(
            b'<?xml version="1.0"?>\n', b'<?xml version="1.0"?>\n<!DOCTYPE root>\n'
        )
        async with fake_router(desc_xml=doctype_xml) as (udp_port, _):
            gw = await discover_gateway(timeout=0.15, ssdp_address=("127.0.0.1", udp_port))
            assert gw is None

        # Lowercase doctype inserted into otherwise valid desc XML
        doctype_lower = DEFAULT_DESC_XML.replace(
            b'<?xml version="1.0"?>\n', b'<?xml version="1.0"?>\n<!doctype root>\n'
        )
        async with fake_router(desc_xml=doctype_lower) as (udp_port, _):
            gw = await discover_gateway(timeout=0.15, ssdp_address=("127.0.0.1", udp_port))
            assert gw is None

        # Internal ENTITY subset inserted into otherwise valid desc XML
        entity_xml = DEFAULT_DESC_XML.replace(
            b'<?xml version="1.0"?>\n',
            b'<?xml version="1.0"?>\n<!DOCTYPE root [ <!ENTITY test "val"> ]>\n',
        )
        async with fake_router(desc_xml=entity_xml) as (udp_port, _):
            gw = await discover_gateway(timeout=0.15, ssdp_address=("127.0.0.1", udp_port))
            assert gw is None

    run(_test())


def test_discover_gateway_refuse_utf16_doctype() -> None:
    async def _test() -> None:
        xml_text = (
            '<?xml version="1.0" encoding="utf-16"?>\n'
            '<!DOCTYPE root [ <!ENTITY test "val"> ]>\n'
            '<root xmlns="urn:schemas-upnp-org:device-1-0">\n'
            "  <device>\n"
            "    <deviceType>urn:schemas-upnp-org:device:InternetGatewayDevice:1</deviceType>\n"
            "    <serviceList>\n"
            "      <service>\n"
            "        <serviceType>urn:schemas-upnp-org:service:WANIPConnection:1</serviceType>\n"
            "        <controlURL>/control</controlURL>\n"
            "      </service>\n"
            "    </serviceList>\n"
            "  </device>\n"
            "</root>\n"
        )
        # UTF-16 with BOM
        utf16_bom = xml_text.encode("utf-16")
        async with fake_router(desc_xml=utf16_bom) as (udp_port, _):
            gw = await discover_gateway(timeout=0.15, ssdp_address=("127.0.0.1", udp_port))
            assert gw is None

        # UTF-16 LE without BOM
        utf16_nobom = xml_text.encode("utf-16-le")
        async with fake_router(desc_xml=utf16_nobom) as (udp_port, _):
            gw = await discover_gateway(timeout=0.15, ssdp_address=("127.0.0.1", udp_port))
            assert gw is None

    run(_test())


def test_discover_gateway_location_different_host_ignored() -> None:
    async def _test() -> None:
        server2 = FakeRouterServer(("127.0.0.2", 0), FakeRouterHTTPHandler)
        thread2 = threading.Thread(
            target=server2.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        )
        thread2.start()
        port2 = server2.server_address[1]

        try:
            custom_loc = f"http://127.0.0.2:{port2}/desc.xml"
            async with fake_router(custom_location=custom_loc) as (udp_port, state1):
                gw = await discover_gateway(timeout=0.15, ssdp_address=("127.0.0.1", udp_port))
                assert gw is None
                assert server2.state.desc_requests == 0
                assert state1.desc_requests == 0
        finally:
            server2.shutdown()
            server2.server_close()
            thread2.join(timeout=1.0)

    run(_test())


def test_discover_gateway_control_url_validation() -> None:
    async def _test() -> None:
        diff_host_xml = DEFAULT_DESC_XML.replace(
            b"<controlURL>/control</controlURL>",
            b"<controlURL>http://127.0.0.2:8000/control</controlURL>",
        )
        async with fake_router(desc_xml=diff_host_xml) as (udp_port, _):
            gw = await discover_gateway(timeout=0.15, ssdp_address=("127.0.0.1", udp_port))
            assert gw is None

        https_xml = DEFAULT_DESC_XML.replace(
            b"<controlURL>/control</controlURL>",
            b"<controlURL>https://127.0.0.1:8000/control</controlURL>",
        )
        async with fake_router(desc_xml=https_xml) as (udp_port, _):
            gw = await discover_gateway(timeout=0.15, ssdp_address=("127.0.0.1", udp_port))
            assert gw is None

    run(_test())


def test_discover_gateway_redirect_not_followed() -> None:
    async def _test() -> None:
        server2 = FakeRouterServer(("127.0.0.2", 0), FakeRouterHTTPHandler)
        thread2 = threading.Thread(
            target=server2.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        )
        thread2.start()
        port2 = server2.server_address[1]

        try:
            redir_target = f"http://127.0.0.2:{port2}/evil"
            async with fake_router(redirect_location=redir_target) as (udp_port, state1):
                gw = await discover_gateway(timeout=0.2, ssdp_address=("127.0.0.1", udp_port))
                assert gw is None
                assert state1.desc_requests == 1
                assert server2.state.desc_requests == 0
        finally:
            server2.shutdown()
            server2.server_close()
            thread2.join(timeout=1.0)

    run(_test())


def test_discover_gateway_non_200_ssdp_reply() -> None:
    async def _test() -> None:
        async with fake_router(ssdp_status="HTTP/1.1 404 Not Found") as (udp_port, state):
            gw = await discover_gateway(timeout=0.15, ssdp_address=("127.0.0.1", udp_port))
            assert gw is None
            assert state.desc_requests == 0

    run(_test())


def test_discover_gateway_oversize_description() -> None:
    async def _test() -> None:
        large_xml = DEFAULT_DESC_XML + b"<!-- " + b"x" * (MAX_RESPONSE_BYTES + 500) + b" -->\n"
        async with fake_router(desc_xml=large_xml) as (udp_port, _):
            gw = await discover_gateway(timeout=0.15, ssdp_address=("127.0.0.1", udp_port))
            assert gw is None

    run(_test())


def test_discover_gateway_oversize_no_content_length() -> None:
    async def _test() -> None:
        large_xml = DEFAULT_DESC_XML + b"<!-- " + b"x" * (MAX_RESPONSE_BYTES + 500) + b" -->\n"
        async with fake_router(desc_xml=large_xml, omit_content_length=True) as (udp_port, _):
            gw = await discover_gateway(timeout=0.2, ssdp_address=("127.0.0.1", udp_port))
            assert gw is None

    run(_test())


def test_add_port_mapping_timeout_and_error() -> None:
    async def _test() -> None:
        async with fake_router(control_delay=1.0) as (udp_port, _):
            gw = await discover_gateway(timeout=0.5, ssdp_address=("127.0.0.1", udp_port))
            assert gw is not None
            t0 = time.monotonic()
            ok = await add_port_mapping(gw, external_port=5000, internal_port=5000, timeout=0.1)
            elapsed = time.monotonic() - t0
            assert ok is False
            assert elapsed < 0.5

        unreachable_gw = Gateway(
            control_url="http://127.0.0.1:1/control",
            service_type="urn:schemas-upnp-org:service:WANIPConnection:1",
            local_ip="127.0.0.1",
        )
        ok = await add_port_mapping(
            unreachable_gw, external_port=5000, internal_port=5000, timeout=0.1
        )
        assert ok is False

    run(_test())


def test_discover_gateway_urlbase_resolution() -> None:
    async def _test() -> None:
        # Desc XML with URLBase and relative controlURL
        urlbase_xml = (
            b'<?xml version="1.0"?>\n'
            b'<root xmlns="urn:schemas-upnp-org:device-1-0">\n'
            b"  <URLBase>http://127.0.0.1:8888/custom_root/</URLBase>\n"
            b"  <device>\n"
            b"    <deviceType>urn:schemas-upnp-org:device:InternetGatewayDevice:1</deviceType>\n"
            b"    <serviceList>\n"
            b"      <service>\n"
            b"        <serviceType>urn:schemas-upnp-org:service:WANIPConnection:1</serviceType>\n"
            b"        <controlURL>rel_control</controlURL>\n"
            b"      </service>\n"
            b"    </serviceList>\n"
            b"  </device>\n"
            b"</root>\n"
        )
        async with fake_router(desc_xml=urlbase_xml) as (udp_port, state):
            # Update URLBase in XML to use actual port so same-host check passes
            real_urlbase = f"http://127.0.0.1:{state.http_port}/custom_root/".encode()
            state.desc_xml = urlbase_xml.replace(
                b"http://127.0.0.1:8888/custom_root/", real_urlbase
            )
            gw = await discover_gateway(timeout=0.2, ssdp_address=("127.0.0.1", udp_port))
            assert gw is not None
            expected_ctrl = f"http://127.0.0.1:{state.http_port}/custom_root/rel_control"
            assert gw.control_url == expected_ctrl

    run(_test())


def test_add_port_mapping_xml_escaping() -> None:
    async def _test() -> None:
        async with fake_router() as (udp_port, state):
            gw = await discover_gateway(timeout=1.0, ssdp_address=("127.0.0.1", udp_port))
            assert gw is not None
            ok = await add_port_mapping(
                gw,
                external_port=5000,
                internal_port=5000,
                description="a<&>b",
                timeout=1.0,
            )
            assert ok is True

            add_posts = [
                p
                for p in state.recorded_posts
                if 'AddPortMapping"' in p["headers"].get("SOAPAction", "")
            ]
            assert len(add_posts) == 1
            body = add_posts[0]["body"]
            assert "<NewPortMappingDescription>a&lt;&amp;&gt;b</NewPortMappingDescription>" in body

    run(_test())


def test_delete_port_mapping_posts_right_action() -> None:
    async def _test() -> None:
        async with fake_router() as (udp_port, state):
            gw = await discover_gateway(timeout=1.0, ssdp_address=("127.0.0.1", udp_port))
            assert gw is not None
            ok = await delete_port_mapping(gw, external_port=5000)
            assert ok is True

            delete_posts = [
                p
                for p in state.recorded_posts
                if 'DeletePortMapping"' in p["headers"].get("SOAPAction", "")
            ]
            assert len(delete_posts) == 1
            body = delete_posts[0]["body"]
            assert "<NewExternalPort>5000</NewExternalPort>" in body
            assert "<NewProtocol>TCP</NewProtocol>" in body
            assert "<NewRemoteHost></NewRemoteHost>" in body
            assert (
                delete_posts[0]["headers"]["SOAPAction"] == f'"{gw.service_type}#DeletePortMapping"'
            )

    run(_test())


def test_external_ip_direct() -> None:
    async def _test() -> None:
        async with fake_router(external_ip="8.8.4.4") as (udp_port, _):
            gw = await discover_gateway(timeout=1.0, ssdp_address=("127.0.0.1", udp_port))
            assert gw is not None
            ip = await external_ip(gw)
            assert ip == "8.8.4.4"

        async with fake_router(external_ip="192.168.1.1") as (udp_port, _):
            gw = await discover_gateway(timeout=1.0, ssdp_address=("127.0.0.1", udp_port))
            assert gw is not None
            ip = await external_ip(gw)
            assert ip is None

    run(_test())


def test_map_port_all_ports_refused() -> None:
    async def _test() -> None:
        refuse = set(range(5000, 5010))
        async with fake_router(refuse_ports=refuse) as (udp_port, _):
            mapping = await map_port(5000, ssdp_address=("127.0.0.1", udp_port))
            assert mapping is None

    run(_test())
