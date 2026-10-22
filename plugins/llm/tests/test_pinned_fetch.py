"""fetch_public_bytes: the one way the bot downloads from a URL a user gave it.

The guarantees under test are what keep a pasted URL from reaching anything
internal, or from stalling a turn: the host is resolved once and the
connection goes to exactly the address that was checked; a redirect is
refused; the size cap and the total deadline hold however the server paces
its bytes.
"""

from __future__ import annotations

import http.server
import socket
import threading
import time
from collections.abc import Iterator

import pytest
from llm.service import fetch_public_bytes

BODY = b"\x89PNG\r\n\x1a\n" + b"x" * 4096


class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_args: object) -> None:
        pass

    def do_GET(self) -> None:
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "http://169.254.169.254/latest/meta-data/")
            self.end_headers()
            return
        if self.path == "/drip":
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.end_headers()
            try:
                for _ in range(200):
                    self.wfile.write(b"x")
                    self.wfile.flush()
                    time.sleep(0.05)
            except OSError:
                pass
            return
        if self.path == "/missing":
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.end_headers()
        self.wfile.write(BODY)


@pytest.fixture
def server() -> Iterator[int]:
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd.server_address[1]
    finally:
        httpd.shutdown()
        httpd.server_close()


def _allow_all(_ip: object) -> bool:
    return True


def test_a_public_image_downloads(server: int) -> None:
    got = fetch_public_bytes(
        f"http://img.test:{server}/cat.png",
        max_bytes=10_000,
        deadline_s=5,
        allow=_allow_all,
        resolve=lambda _host: ["127.0.0.1"],
    )
    assert got == (BODY, "image/png")


def test_a_host_with_any_disallowed_address_is_never_contacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connects: list[object] = []
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: connects.append(a))
    got = fetch_public_bytes(
        "http://rebind.test/cat.png",
        max_bytes=10_000,
        deadline_s=5,
        resolve=lambda _host: ["93.184.216.34", "10.0.0.5"],
    )
    assert got is None
    assert connects == []


def test_the_connection_goes_to_the_address_that_was_checked(server: int) -> None:
    """DNS rebinding: the name resolves once. A second lookup that would
    have answered with an internal address is never made."""
    answers = iter([["127.0.0.1"], ["10.255.255.1"]])
    lookups: list[str] = []

    def resolve(host: str) -> list[str]:
        lookups.append(host)
        return next(answers)

    got = fetch_public_bytes(
        f"http://img.test:{server}/cat.png",
        max_bytes=10_000,
        deadline_s=5,
        allow=_allow_all,
        resolve=resolve,
    )
    assert got == (BODY, "image/png")
    assert lookups == ["img.test"]


def test_the_default_policy_refuses_loopback(server: int) -> None:
    got = fetch_public_bytes(
        f"http://img.test:{server}/cat.png",
        max_bytes=10_000,
        deadline_s=5,
        resolve=lambda _host: ["127.0.0.1"],
    )
    assert got is None


@pytest.mark.parametrize("path", ["/redirect", "/missing"])
def test_anything_but_200_is_refused(server: int, path: str) -> None:
    got = fetch_public_bytes(
        f"http://img.test:{server}{path}",
        max_bytes=10_000,
        deadline_s=5,
        allow=_allow_all,
        resolve=lambda _host: ["127.0.0.1"],
    )
    assert got is None


def test_a_body_over_the_cap_is_refused(server: int) -> None:
    got = fetch_public_bytes(
        f"http://img.test:{server}/cat.png",
        max_bytes=len(BODY) - 1,
        deadline_s=5,
        allow=_allow_all,
        resolve=lambda _host: ["127.0.0.1"],
    )
    assert got is None


def test_a_slow_drip_cannot_outlast_the_deadline(server: int) -> None:
    """Each byte arrives well inside any per-read timeout; only a total
    deadline stops a server that drips."""
    start = time.monotonic()
    got = fetch_public_bytes(
        f"http://img.test:{server}/drip",
        max_bytes=10_000,
        deadline_s=0.5,
        allow=_allow_all,
        resolve=lambda _host: ["127.0.0.1"],
    )
    assert got is None
    assert time.monotonic() - start < 2


@pytest.mark.parametrize(
    "url", ["ftp://img.test/cat.png", "file:///etc/passwd", "http://[/x", "http:///x"]
)
def test_only_http_urls_with_a_host_are_fetched(url: str) -> None:
    assert fetch_public_bytes(url, max_bytes=10, deadline_s=1, allow=_allow_all) is None
