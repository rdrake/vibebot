"""Uploads to the network's file host (IRCv3 draft/FILEHOST + draft/authtoken).

Expected values come from what FractalRealities.AfterNET.Org and
paste.boxlabs.uk sent a logged-in client on 2026-11-05: a one-line
``TOKEN GENERATE FILEHOST :<jwt>``, then ``201 Created`` with a ``Location``.
"""

from __future__ import annotations

import http.server
import threading
from typing import TYPE_CHECKING, ClassVar

import pytest
from llm.assistant import ToolCallbackResult
from llm.service import _is_image_failure
from supybot import ircmsgs

from .conftest import make_completion_response, make_tool_call

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pytest_mock import MockerFixture

ENDPOINT = "https://paste.boxlabs.uk/filehost"
LINK = "https://paste.boxlabs.uk/filehost/LcznV0hJYMGeojXa81XYxe.png"
# Not a PNG header, so _save_image_bytes sends these bytes as they are.
IMAGE = b"\xff\xd8\xffjpeg bytes"


def _token_reply(token: str) -> ircmsgs.IrcMsg:
    return ircmsgs.IrcMsg(command="TOKEN", args=("GENERATE", "FILEHOST", token))


class TestTokenFromTheServer:
    @pytest.fixture
    def env(self, plugin_env, mocker: MockerFixture):
        plugin, irc, _msg = plugin_env
        irc.network = "AfterNET"
        irc.state.capabilities_ack = {"draft/authtoken"}
        irc.state.supported = {"draft/FILEHOST": ENDPOINT}
        mocker.patch("llm.plugin.world.ircs", [irc])
        return plugin, irc

    def test_the_servers_reply_is_the_token(self, env) -> None:
        plugin, irc = env

        def reply(m) -> None:
            assert (m.command, m.args) == ("TOKEN", ("GENERATE", "FILEHOST"))
            plugin.doToken(irc, _token_reply("a.b.c"))

        irc.queueMsg.side_effect = reply
        assert plugin.filehost_token() == "a.b.c"

    def test_a_refusal_ends_the_wait_and_logs_the_reason(self, env, mocker: MockerFixture) -> None:
        plugin, irc = env
        log = mocker.patch.object(plugin, "log")
        mocker.patch.object(plugin, "_IRC_QUERY_TIMEOUT", 30)
        irc.queueMsg.side_effect = lambda _m: plugin.doFail(
            irc, ircmsgs.IrcMsg(command="FAIL", args=("TOKEN", "ACCOUNT_REQUIRED", "Log in"))
        )

        assert plugin.filehost_token() is None
        assert log.warning.call_args[0][1] == "ACCOUNT_REQUIRED: Log in"

    def test_another_commands_failure_is_not_a_refusal(self, env, mocker: MockerFixture) -> None:
        plugin, irc = env
        mocker.patch.object(plugin, "_IRC_QUERY_TIMEOUT", 0.05)

        def reply(_m) -> None:
            plugin.doFail(irc, ircmsgs.IrcMsg(command="FAIL", args=("REDACT", "NOPE", "no")))
            plugin.doToken(irc, _token_reply("still-mine"))

        irc.queueMsg.side_effect = reply
        assert plugin.filehost_token() == "still-mine"

    def test_a_token_for_another_service_is_not_ours(self, env, mocker: MockerFixture) -> None:
        plugin, irc = env
        mocker.patch.object(plugin, "_IRC_QUERY_TIMEOUT", 0.01)
        irc.queueMsg.side_effect = lambda _m: plugin.doToken(
            irc, ircmsgs.IrcMsg(command="TOKEN", args=("GENERATE", "OTHER", "theirs"))
        )
        assert plugin.filehost_token() is None

    def test_a_reply_without_a_token_is_ignored(self, env, mocker: MockerFixture) -> None:
        plugin, irc = env
        mocker.patch.object(plugin, "_IRC_QUERY_TIMEOUT", 0.01)
        irc.queueMsg.side_effect = lambda _m: plugin.doToken(
            irc, ircmsgs.IrcMsg(command="TOKEN", args=("GENERATE", "FILEHOST"))
        )
        assert plugin.filehost_token() is None

    def test_no_reply_gives_up(self, env, mocker: MockerFixture) -> None:
        plugin, _irc = env
        mocker.patch.object(plugin, "_IRC_QUERY_TIMEOUT", 0.01)
        assert plugin.filehost_token() is None

    def test_concurrent_uploads_never_share_a_single_use_token(self, env) -> None:
        plugin, irc = env
        issued = iter(["one", "two"])
        irc.queueMsg.side_effect = lambda _m: plugin.doToken(irc, _token_reply(next(issued)))
        start = threading.Barrier(2)
        results: list = []

        def worker() -> None:
            start.wait(5)
            results.append(plugin.filehost_token())

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
        assert sorted(results) == ["one", "two"]

    @pytest.mark.parametrize(
        ("caps", "supported"),
        [
            (set(), {"draft/FILEHOST": ENDPOINT}),
            ({"draft/authtoken"}, {}),
            ({"draft/authtoken"}, {"draft/FILEHOST": "http://paste.boxlabs.uk/filehost"}),
        ],
    )
    def test_a_network_without_a_usable_host_is_never_asked(self, env, caps, supported) -> None:
        plugin, irc = env
        irc.state.capabilities_ack = caps
        irc.state.supported = supported
        assert plugin.filehost_endpoint() is None
        assert plugin.filehost_token() is None
        irc.queueMsg.assert_not_called()

    def test_the_advertised_host_is_the_endpoint(self, env) -> None:
        plugin, _irc = env
        assert plugin.filehost_endpoint() == ENDPOINT


class _Host(http.server.BaseHTTPRequestHandler):
    """One path per behaviour of the upload host."""

    seen: ClassVar[list[tuple[str, dict, bytes]]] = []

    def log_message(self, *_args: object) -> None:
        pass

    def _send(self, status: int, body: bytes = b"", location: str | None = None) -> None:
        self.send_response(status)
        if location:
            self.send_header("Location", location)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        type(self).seen.append((self.path, dict(self.headers), body))
        here = f"http://{self.headers['Host']}"
        replies = {
            "/ok": (201, b"", f"{here}/filehost/abc.jpg"),
            "/relative": (201, b"", "/filehost/abc.jpg"),
            "/no-location": (201, b'{"url": "ignored"}', None),
            "/off-host": (201, b"", "https://evil.example.net/x.jpg"),
            "/not-image": (201, b"", f"{here}/filehost/page.html"),
            "/traversal": (201, b"", f"{here}/filehost/../x.jpg"),
            "/replayed": (401, b'{"message":"This token has already been used"}', None),
            "/moved": (302, b"", f"{here}/stolen"),
        }
        self._send(*replies[self.path])


@pytest.fixture
def host() -> Iterator[str]:
    _Host.seen.clear()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Host)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


class TestSavingAnImage:
    @pytest.fixture
    def service(self, make_service, tmp_path, mocker: MockerFixture):
        service, plugin = make_service(
            httpRoot=str(tmp_path), httpUrlBase="https://example.com/llm", timeout=30
        )
        # The local test host is loopback, which the real check rightly refuses.
        mocker.patch("llm.service.validate_external_url", return_value=True)
        plugin.filehost_token = lambda: "a.b.c"
        return service

    def _save(self, service, url: str) -> str:
        service.plugin.filehost_endpoint = lambda: url
        return service._save_image_bytes(IMAGE, "jpg")

    def test_the_file_goes_up_raw_with_the_token_as_bearer(self, service, host, tmp_path) -> None:
        assert self._save(service, f"{host}/ok") == f"{host}/filehost/abc.jpg"

        _path, headers, body = _Host.seen[0]
        assert body == IMAGE
        assert headers["Authorization"] == "Bearer a.b.c"
        assert headers["Content-Type"] == "image/jpeg"
        assert list(tmp_path.iterdir()) == []

    def test_a_relative_location_resolves_against_the_endpoint(self, service, host) -> None:
        assert self._save(service, f"{host}/relative") == f"{host}/filehost/abc.jpg"

    @pytest.mark.parametrize(
        "path", ["/no-location", "/off-host", "/not-image", "/traversal", "/replayed"]
    )
    def test_anything_but_an_image_on_the_host_stores_locally(self, service, host, path) -> None:
        assert self._save(service, f"{host}{path}").startswith("https://example.com/llm/img_")

    def test_a_refusal_logs_the_hosts_reason(self, service, host, mocker: MockerFixture) -> None:
        log = mocker.patch.object(service, "log")
        self._save(service, f"{host}/replayed")
        assert "already been used" in log.warning.call_args[0][3]

    def test_a_redirect_never_carries_the_token_onward(self, service, host) -> None:
        assert self._save(service, f"{host}/moved").startswith("https://example.com/llm/img_")
        assert [path for path, _headers, _body in _Host.seen] == ["/moved"]

    def test_an_unreachable_host_stores_locally(self, service) -> None:
        result = self._save(service, "http://127.0.0.1:9/ok")
        assert result.startswith("https://example.com/llm/img_")

    def test_no_token_never_contacts_the_host(self, service, host) -> None:
        service.plugin.filehost_token = lambda: None
        assert self._save(service, f"{host}/ok").startswith("https://example.com/llm/img_")
        assert _Host.seen == []


class TestAnUnsafeHost:
    def test_is_never_given_a_token(self, make_service, tmp_path, mocker: MockerFixture) -> None:
        service, plugin = make_service(httpRoot=str(tmp_path), httpUrlBase="https://e.example/llm")
        plugin.filehost_endpoint = lambda: "https://127.0.0.1/filehost"
        token = mocker.Mock()
        plugin.filehost_token = token

        service._save_image_bytes(IMAGE, "jpg")

        token.assert_not_called()


class TestHostCountsAsOurs:
    @pytest.fixture
    def service(self, make_service):
        service, plugin = make_service(httpUrlBase="https://bot.example/llm")
        plugin.filehost_endpoint = lambda: ENDPOINT
        return service

    def test_for_the_fabrication_guard(self, service) -> None:
        assert service.own_image_hosts() == {"paste.boxlabs.uk", "bot.example"}

    def test_pages_keep_its_illustrations_and_drop_other_images(self, service) -> None:
        html = f'<img src="{LINK}"><img src="img_local.png"><img src="https://evil.example/x.png">'
        assert service._restrict_img_srcs(html, "https://bot.example/llm") == (
            f'<img src="{LINK}"><img src="img_local.png">'
        )

    def test_pages_drop_its_images_when_the_network_has_no_host(self, service) -> None:
        service.plugin.filehost_endpoint = lambda: None
        html = f'<img src="{LINK}">'
        assert service._restrict_img_srcs(html, "https://bot.example/llm") == ""

    def test_pages_embed_its_links_absolutely(self, service) -> None:
        assert service._page_image_ref(LINK) == LINK


class TestGuardsAcceptItsLinks:
    """A file host link has no img_ filename. Delivered is still delivered."""

    def test_a_delivered_link_is_not_a_failure_report(self) -> None:
        assert not _is_image_failure(f"First attempt failed, here you go: {LINK}")

    def test_a_draw_that_returned_its_link_reaches_the_channel(
        self, make_service, mocker: MockerFixture
    ) -> None:
        """A step mixing a draw with another tool reaches the guard with the URL
        in the model's text, and an unrecognised URL is replaced by the failure
        message."""
        service, plugin = make_service(
            assistantModel="xai/grok-4.3", httpUrlBase="https://irc.rdrake.org/llm"
        )
        plugin.filehost_endpoint = lambda: ENDPOINT
        mocker.patch(
            "llm.service.provider.chat",
            side_effect=[
                make_completion_response(
                    None,
                    tool_calls=[
                        make_tool_call("generate_image", {"prompt": "a wizard"}),
                        make_tool_call("no_such_tool", {}),
                    ],
                ),
                make_completion_response(f"Here you go: {LINK}"),
            ],
        )

        result = service.assistant_completion(
            prompt="draw a wizard",
            nick="ibutsu",
            channel="#afternet",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            capabilities=frozenset({"llm.ask", "llm.draw"}),
            account="ibutsu",
            draw_fn=lambda _p, **_kw: ToolCallbackResult(True, LINK),
        )

        assert LINK in (result.content or "")
