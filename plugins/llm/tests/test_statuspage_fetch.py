"""SSRF and resource guards on the one place this feature opens a socket."""

from __future__ import annotations

import io
import json

import pytest
from llm import statuspage


class FakeResponse(io.BytesIO):
    def __init__(self, body: bytes, headers: dict[str, str], status: int = 200):
        super().__init__(body)
        self.headers = headers
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
        return False


class FakeOpener:
    """Records the Request it was given and returns a canned response."""

    def __init__(self, response=None, raises=None):
        self.response = response
        self.raises = raises
        self.request = None

    def open(self, req, timeout=None):  # noqa: ARG002
        self.request = req
        if self.raises:
            raise self.raises
        return self.response


def good_body() -> bytes:
    return json.dumps(
        {
            "page": {"name": "Claude", "url": "https://status.claude.com"},
            "status": {"indicator": "none", "description": "All Systems Operational"},
            "components": [],
            "incidents": [],
            "scheduled_maintenances": [],
        }
    ).encode()


def call(opener, *, etag=None, modified=None, validate=None, resolves=None):
    return statuspage.fetch_summary(
        "https://status.claude.com",
        timeout=10,
        etag=etag,
        modified=modified,
        validate=validate if validate is not None else (lambda _u: True),
        resolves_public=resolves if resolves is not None else (lambda _u: True),
        opener_factory=lambda: opener,
    )


class TestSsrfGuards:
    def test_refuses_when_validate_rejects(self):
        opener = FakeOpener(FakeResponse(good_body(), {"Content-Type": "application/json"}))
        with pytest.raises(statuspage.FetchError, match="rejected"):
            call(opener, validate=lambda _u: False)
        assert opener.request is None, "must fail before opening a socket"

    def test_refuses_when_host_is_not_globally_routable(self):
        opener = FakeOpener(FakeResponse(good_body(), {"Content-Type": "application/json"}))
        with pytest.raises(statuspage.FetchError, match="public"):
            call(opener, resolves=lambda _u: False)
        assert opener.request is None

    def test_builds_url_from_base_without_letting_input_into_the_path(self):
        opener = FakeOpener(FakeResponse(good_body(), {"Content-Type": "application/json"}))
        call(opener)
        assert opener.request.full_url == "https://status.claude.com/api/v2/summary.json"

    def test_trailing_slash_on_base_does_not_double(self):
        opener = FakeOpener(FakeResponse(good_body(), {"Content-Type": "application/json"}))
        statuspage.fetch_summary(
            "https://status.claude.com/",
            timeout=10,
            validate=lambda _u: True,
            resolves_public=lambda _u: True,
            opener_factory=lambda: opener,
        )
        assert opener.request.full_url == "https://status.claude.com/api/v2/summary.json"

    def test_validate_raising_becomes_fetch_error(self):
        opener = FakeOpener(FakeResponse(good_body(), {"Content-Type": "application/json"}))

        def boom(_u):
            raise UnicodeEncodeError("idna", "x", 0, 1, "bad label")

        with pytest.raises(statuspage.FetchError):
            call(opener, validate=boom)
        assert opener.request is None

    def test_resolves_public_raising_becomes_fetch_error(self):
        opener = FakeOpener(FakeResponse(good_body(), {"Content-Type": "application/json"}))

        def boom(_u):
            raise UnicodeEncodeError("idna", "x", 0, 1, "bad label")

        with pytest.raises(statuspage.FetchError):
            call(opener, resolves=boom)
        assert opener.request is None


class TestBaseUrlValidation:
    """statusPageUrl is operator config, but a misconfiguration (a path,
    query, fragment, or non-http scheme) must fail loudly rather than be
    silently mis-fetched by naive string concatenation."""

    @pytest.mark.parametrize(
        "base_url",
        [
            "https://status.claude.com/some/path",
            "https://status.claude.com?x=1",
            "https://status.claude.com#frag",
            "ftp://status.claude.com",
            "javascript://status.claude.com",
        ],
    )
    def test_malformed_base_url_raises_and_opens_no_socket(self, base_url):
        opener = FakeOpener(FakeResponse(good_body(), {"Content-Type": "application/json"}))
        with pytest.raises(statuspage.FetchError):
            statuspage.fetch_summary(
                base_url,
                timeout=10,
                validate=lambda _u: True,
                resolves_public=lambda _u: True,
                opener_factory=lambda: opener,
            )
        assert opener.request is None


class TestResponseGuards:
    def test_rejects_non_json_content_type(self):
        opener = FakeOpener(FakeResponse(b"<html></html>", {"Content-Type": "text/html"}))
        with pytest.raises(statuspage.FetchError, match="content-type"):
            call(opener)

    def test_rejects_oversize_body(self):
        big = b"x" * (statuspage.MAX_RESPONSE_BYTES + 10)
        opener = FakeOpener(FakeResponse(big, {"Content-Type": "application/json"}))
        with pytest.raises(statuspage.FetchError, match="too large"):
            call(opener)

    def test_rejects_undecodable_json(self):
        opener = FakeOpener(FakeResponse(b"{not json", {"Content-Type": "application/json"}))
        with pytest.raises(statuspage.FetchError, match="JSON"):
            call(opener)

    def test_network_error_becomes_fetch_error(self):
        opener = FakeOpener(raises=OSError("connection refused"))
        with pytest.raises(statuspage.FetchError):
            call(opener)


class TestConditionalGet:
    def test_sends_validators_when_known(self):
        opener = FakeOpener(FakeResponse(good_body(), {"Content-Type": "application/json"}))
        call(opener, etag='W/"abc"', modified="Sat, 09 Aug 2026 14:00:00 GMT")
        assert opener.request.get_header("If-none-match") == 'W/"abc"'
        assert opener.request.get_header("If-modified-since") == "Sat, 09 Aug 2026 14:00:00 GMT"

    def test_omits_validators_when_unknown(self):
        opener = FakeOpener(FakeResponse(good_body(), {"Content-Type": "application/json"}))
        call(opener)
        assert opener.request.get_header("If-none-match") is None

    def test_304_returns_not_modified_with_no_payload(self):
        import urllib.error

        err = urllib.error.HTTPError(
            "https://status.claude.com/api/v2/summary.json", 304, "Not Modified", {}, None
        )
        opener = FakeOpener(raises=err)
        result = call(opener, etag='W/"abc"')
        assert result.not_modified is True
        assert result.payload is None

    def test_returns_validators_from_the_response(self):
        opener = FakeOpener(
            FakeResponse(
                good_body(),
                {
                    "Content-Type": "application/json",
                    "ETag": 'W/"new"',
                    "Last-Modified": "Sat, 09 Aug 2026 15:00:00 GMT",
                },
            )
        )
        result = call(opener)
        assert result.not_modified is False
        assert result.etag == 'W/"new"'
        assert result.modified == "Sat, 09 Aug 2026 15:00:00 GMT"
        assert result.payload["page"]["name"] == "Claude"


class TestRealOpenerRefusesRedirects:
    """Exercises _default_opener_factory itself, not a FakeOpener.

    Redirect refusal is the canonical SSRF escape: a 302 to a link-local
    address would otherwise land instance metadata in the poller cache and
    get announced to a channel.
    """

    @staticmethod
    def _serve(handler_cls):
        import http.server
        import threading

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        return srv, srv.server_address[1]

    def test_302_to_link_local_raises_fetch_error(self):
        import http.server

        class Redirector(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def do_GET(self):  # noqa: N802
                self.send_response(302)
                self.send_header("Location", "http://169.254.169.254/latest/meta-data/")
                self.end_headers()

            def log_message(self, *_args):
                pass

        srv, port = self._serve(Redirector)
        try:
            with pytest.raises(statuspage.FetchError):
                statuspage.fetch_summary(
                    f"http://127.0.0.1:{port}",
                    timeout=5,
                    validate=lambda _u: True,
                    resolves_public=lambda _u: True,
                )
        finally:
            srv.shutdown()
            srv.server_close()

    def test_real_opener_still_fetches_a_normal_200(self):
        """Proves the redirect test above is not just 'everything fails'."""
        import http.server

        body = good_body()

        class Server(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def do_GET(self):  # noqa: N802
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args):
                pass

        srv, port = self._serve(Server)
        try:
            result = statuspage.fetch_summary(
                f"http://127.0.0.1:{port}",
                timeout=5,
                validate=lambda _u: True,
                resolves_public=lambda _u: True,
            )
        finally:
            srv.shutdown()
            srv.server_close()
        assert result.not_modified is False
        assert result.payload["page"]["name"] == "Claude"
