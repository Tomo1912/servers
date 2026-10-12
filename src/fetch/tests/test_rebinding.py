"""Tests for the DNS rebinding half of the private-address guard.

The request hook resolves a host before each request, and httpcore used to
resolve it again when it connected, so a DNS server that answers with a public
address first and an internal one second got past the hook. These tests use
the real httpx client (no ``web`` fixture) against a real HTTP server on
127.0.0.1, with only the server's resolver replaced, so what is shown is that
the connection itself is refused, not just the request. The remaining tests
drive ``_PublicAddressBackend`` directly, with a fake inner backend, for the
cases a local socket cannot produce.
"""

from __future__ import annotations

import socket
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterable

import httpcore
import httpx
import pytest
from mcp.shared.exceptions import MCPError
from mcp.types import INTERNAL_ERROR, INVALID_PARAMS

import mcp_server_fetch.server as server_module
from mcp_server_fetch.server import (
    _install_public_address_backend,
    _PublicAddressBackend,
)

from .harness import PUBLIC_IP, connect

OTHER_PUBLIC_IP = "93.184.215.15"


class LocalServer:
    """A plain-text HTTP server on 127.0.0.1 that records each path it serves."""

    def __init__(self) -> None:
        self.paths: list[str] = []
        paths = self.paths

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                paths.append(self.path)
                body = b"internal secret"
                self.send_response(200)
                self.send_header("content-type", "text/plain")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:
                pass  # Keep the request log off the console.

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]


@pytest.fixture
def local_server() -> Iterator[LocalServer]:
    server = LocalServer()
    thread = threading.Thread(target=server.httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.httpd.shutdown()
        server.httpd.server_close()
        thread.join()


def rebinding_resolver(*answers: list[str]) -> Any:
    """A resolver that gives each answer in turn, then repeats the last one."""
    calls: list[str] = []

    async def resolve(host: str) -> list[str]:
        answer = answers[min(len(calls), len(answers) - 1)]
        calls.append(host)
        return answer

    resolve.calls = calls  # type: ignore[attr-defined]
    return resolve


def text_of(result: Any) -> str:
    data = result.model_dump(by_alias=True, mode="json")
    assert len(data["content"]) == 1
    return data["content"][0]["text"]


async def test_rebinding_to_loopback_is_refused_at_connect(
    monkeypatch: pytest.MonkeyPatch, local_server: LocalServer
) -> None:
    # The hook sees a public address; the connection-time lookup gets
    # 127.0.0.1, where a real server is listening. Without the backend, httpx
    # would connect to localhost and read the page.
    resolve = rebinding_resolver([PUBLIC_IP], ["127.0.0.1"])
    monkeypatch.setattr(server_module, "_resolve_host", resolve)
    url = f"http://localhost:{local_server.port}/admin"
    async with connect(ignore_robots_txt=True) as (session, _):
        result = await session.call_tool("fetch", {"url": url})
    assert result.is_error is True
    assert text_of(result) == (
        f"Refused to fetch {url}: localhost resolves to 127.0.0.1, which is not a "
        "public address. Start the server with --allow-private-ips to allow "
        "private, loopback and link-local addresses."
    )
    # Resolved once by the hook and once by the backend, never by httpcore.
    assert resolve.calls == ["localhost", "localhost"]
    assert local_server.paths == []


async def test_allowed_private_host_reaches_the_local_server(
    local_server: LocalServer,
) -> None:
    # The control: the same server is reachable once localhost is allowed,
    # with the real resolver and the backend installed.
    url = f"http://localhost:{local_server.port}/admin"
    async with connect(ignore_robots_txt=True, allow_private_hosts=["localhost"]) as (
        session,
        _,
    ):
        result = await session.call_tool("fetch", {"url": url})
    assert result.is_error is False
    assert text_of(result).endswith(f"Contents of {url}:\ninternal secret")
    assert local_server.paths == ["/admin"]


async def test_loopback_is_refused_without_the_allowlist(
    local_server: LocalServer,
) -> None:
    url = f"http://localhost:{local_server.port}/admin"
    async with connect(ignore_robots_txt=True) as (session, _):
        result = await session.call_tool("fetch", {"url": url})
    assert result.is_error is True
    assert local_server.paths == []


# --------------------------------------------------------------------------
# _PublicAddressBackend, with a fake inner backend
# --------------------------------------------------------------------------


class FakeStream(httpcore.AsyncNetworkStream):
    def __init__(self, address: str) -> None:
        self.address = address
        self.tls: dict[str, Any] = {}

    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        raise NotImplementedError  # pragma: no cover  # TLS fails first

    async def write(self, buffer: bytes, timeout: float | None = None) -> None:
        raise NotImplementedError  # pragma: no cover  # TLS fails first

    async def aclose(self) -> None:
        pass

    async def start_tls(
        self, ssl_context: Any, server_hostname: str | None = None, timeout: Any = None
    ) -> httpcore.AsyncNetworkStream:
        self.tls["server_hostname"] = server_hostname
        raise httpcore.ConnectError("stop after recording the TLS handshake")


class FakeBackend(httpcore.AsyncNetworkBackend):
    def __init__(self, refuse: Iterable[str] = ()) -> None:
        self.refuse = set(refuse)
        self.attempts: list[tuple[str, int]] = []
        self.streams: list[FakeStream] = []
        self.calls: list[str] = []

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> httpcore.AsyncNetworkStream:
        self.attempts.append((host, port))
        if host in self.refuse:
            raise httpcore.ConnectError(f"refused {host}")
        stream = FakeStream(host)
        self.streams.append(stream)
        return stream

    async def connect_unix_socket(
        self, path: str, timeout: float | None = None, socket_options: Any = None
    ) -> httpcore.AsyncNetworkStream:
        self.calls.append(f"unix {path}")
        return FakeStream(path)

    async def sleep(self, seconds: float) -> None:
        self.calls.append(f"sleep {seconds}")


async def test_backend_dials_the_checked_addresses_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolve = rebinding_resolver([PUBLIC_IP, OTHER_PUBLIC_IP])
    monkeypatch.setattr(server_module, "_resolve_host", resolve)
    inner = FakeBackend(refuse=[PUBLIC_IP])
    backend = _PublicAddressBackend(inner, frozenset())
    stream = await backend.connect_tcp("example.com", 443)
    assert isinstance(stream, FakeStream)
    assert stream.address == OTHER_PUBLIC_IP
    assert inner.attempts == [(PUBLIC_IP, 443), (OTHER_PUBLIC_IP, 443)]
    assert resolve.calls == ["example.com"]


async def test_backend_raises_the_last_error_when_every_address_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        server_module,
        "_resolve_host",
        rebinding_resolver([PUBLIC_IP, OTHER_PUBLIC_IP]),
    )
    inner = FakeBackend(refuse=[PUBLIC_IP, OTHER_PUBLIC_IP])
    backend = _PublicAddressBackend(inner, frozenset())
    with pytest.raises(httpcore.ConnectError, match=f"refused {OTHER_PUBLIC_IP}"):
        await backend.connect_tcp("example.com", 80)


async def test_backend_refuses_when_any_address_is_private(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Called without the hook, the message names the host in place of the URL.
    monkeypatch.setattr(
        server_module, "_resolve_host", rebinding_resolver([PUBLIC_IP, "10.0.0.1"])
    )
    inner = FakeBackend()
    backend = _PublicAddressBackend(inner, frozenset())
    with pytest.raises(MCPError) as excinfo:
        await backend.connect_tcp("example.com", 80)
    assert excinfo.value.error.code == INVALID_PARAMS
    assert str(excinfo.value).startswith(
        "Refused to fetch example.com: example.com resolves to 10.0.0.1, "
    )
    assert inner.attempts == []


async def test_backend_turns_a_failed_lookup_into_a_connect_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def unresolvable(host: str) -> list[str]:
        raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")

    monkeypatch.setattr(server_module, "_resolve_host", unresolvable)
    backend = _PublicAddressBackend(FakeBackend(), frozenset())
    with pytest.raises(httpcore.ConnectError, match="Name or service not known"):
        await backend.connect_tcp("nowhere.example", 80)


async def test_backend_passes_an_allowed_host_through_unresolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolve = rebinding_resolver(["127.0.0.1"])
    monkeypatch.setattr(server_module, "_resolve_host", resolve)
    inner = FakeBackend()
    backend = _PublicAddressBackend(inner, frozenset({"intranet.example.com"}))
    await backend.connect_tcp("Intranet.Example.com.", 80)
    assert inner.attempts == [("Intranet.Example.com.", 80)]
    assert resolve.calls == []


async def test_backend_delegates_unix_sockets_and_sleep() -> None:
    inner = FakeBackend()
    backend = _PublicAddressBackend(inner, frozenset())
    await backend.connect_unix_socket("/tmp/socket")
    await backend.sleep(0)
    assert inner.calls == ["unix /tmp/socket", "sleep 0"]


async def test_tls_and_sni_keep_the_host_name(monkeypatch: pytest.MonkeyPatch) -> None:
    # The TCP connection goes to the checked address; the TLS handshake (and so
    # SNI and certificate verification) still names the host.
    monkeypatch.setattr(server_module, "_resolve_host", rebinding_resolver([PUBLIC_IP]))
    async with httpx.AsyncClient() as client:
        _install_public_address_backend(client, frozenset())
        backend = client._transport._pool._network_backend  # type: ignore[attr-defined]
        assert isinstance(backend, _PublicAddressBackend)
        inner = FakeBackend()
        backend._inner = inner
        with pytest.raises(httpx.ConnectError):
            await client.get("https://example.com/")
    assert inner.attempts == [(PUBLIC_IP, 443)]
    assert inner.streams[0].tls == {"server_hostname": "example.com"}


async def test_install_fails_when_httpx_internals_change() -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200))
    ) as client:
        with pytest.raises(MCPError) as excinfo:
            _install_public_address_backend(client, frozenset())
    assert excinfo.value.error.code == INTERNAL_ERROR
    assert "does not expose the connection pool" in str(excinfo.value)


async def test_install_leaves_proxy_transports_alone() -> None:
    async with httpx.AsyncClient(proxy="http://proxy.example.com:8080") as client:
        _install_public_address_backend(client, frozenset())
        direct = client._transport._pool._network_backend  # type: ignore[attr-defined]
        proxies = [
            mount._pool._network_backend  # type: ignore[union-attr]
            for mount in client._mounts.values()
        ]
    assert isinstance(direct, _PublicAddressBackend)
    assert proxies
    assert not any(isinstance(b, _PublicAddressBackend) for b in proxies)
