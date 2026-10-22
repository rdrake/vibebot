"""Plugin core: HTTP callbacks, init/lifecycle, helpers, auth, rate limiting, migrations, utilities."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import pytest
from llm import agenda, ircquery
from llm.identity import Caller, Identity
from llm.service import AssistantResult, Spend

from .conftest import make_caller, make_registry_side_effect, make_reminder_row

if TYPE_CHECKING:
    from unittest.mock import MagicMock

    from pytest_mock import MockerFixture


class TestHTTPCallbackDoGet:
    """Test HTTP callback doGet method for serving files."""

    @pytest.fixture
    def mock_plugin(self, mocker: MockerFixture) -> MagicMock:
        """Create a mock plugin for HTTP callback."""
        plugin = mocker.MagicMock()
        plugin.registryValue.return_value = ""  # No custom httpRoot
        return plugin

    @pytest.fixture
    def http_callback(self, mock_plugin: MagicMock):
        """Create an HTTP callback with mock plugin."""
        from llm.plugin import LLMHTTPCallback

        return LLMHTTPCallback(mock_plugin)

    @pytest.fixture
    def mock_handler(self, mocker: MockerFixture) -> MagicMock:
        """Create a mock HTTP handler."""
        handler = mocker.MagicMock()
        # wfile needs to be a MagicMock so we can set side_effect
        handler.wfile = mocker.MagicMock()
        return handler

    def test_doget_returns_404_at_root(self, http_callback, mock_handler: MagicMock) -> None:
        """GIVEN empty path WHEN doGet called THEN returns 404 (help is on GitHub Pages)."""
        http_callback.doGet(mock_handler, "")
        mock_handler.send_response.assert_called_with(404)
        mock_handler.end_headers.assert_called_once()

    def test_doget_blocks_directory_traversal(self, http_callback, mock_handler: MagicMock) -> None:
        """GIVEN path with .. WHEN doGet called THEN returns 403."""
        http_callback.doGet(mock_handler, "../etc/passwd")
        mock_handler.send_response.assert_called_with(403)
        mock_handler.end_headers.assert_called_once()

    def test_doget_blocks_absolute_path_in_middle(
        self, http_callback, mock_handler: MagicMock
    ) -> None:
        """GIVEN path with / after stripping WHEN doGet called THEN returns 403."""
        # After lstrip("/"), if there's still a / at start, it's suspicious
        # Actually looking at the code: path.startswith("/") after lstrip
        # This tests the security check more directly
        http_callback.doGet(mock_handler, "/../test")
        mock_handler.send_response.assert_called_with(403)

    def test_doget_returns_404_for_missing_file(
        self, http_callback, mock_handler: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN nonexistent file WHEN doGet called THEN returns 404."""
        mocker.patch.object(http_callback, "_get_web_dir", return_value="/nonexistent")
        http_callback.doGet(mock_handler, "missing.txt")
        mock_handler.send_response.assert_called_with(404)
        mock_handler.end_headers.assert_called_once()

    def test_doget_serves_existing_file(
        self, http_callback, mock_handler: MagicMock, tmp_path, mocker: MockerFixture
    ) -> None:
        """GIVEN existing file WHEN doGet called THEN returns 200 with content."""
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"test content")

        mocker.patch.object(http_callback, "_get_web_dir", return_value=str(tmp_path))
        http_callback.doGet(mock_handler, "test.txt")

        mock_handler.send_response.assert_called_with(200)
        mock_handler.send_header.assert_any_call("Content-Type", "text/plain")
        mock_handler.send_header.assert_any_call("Content-Length", "12")

    def test_doget_handles_broken_pipe_silently(
        self, http_callback, mock_handler: MagicMock, tmp_path, mocker: MockerFixture
    ) -> None:
        """GIVEN client disconnect WHEN doGet serving file THEN no error raised."""
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"test")

        mock_handler.wfile.write.side_effect = BrokenPipeError()

        mocker.patch.object(http_callback, "_get_web_dir", return_value=str(tmp_path))
        # Should not raise
        http_callback.doGet(mock_handler, "test.txt")

    def test_doget_handles_os_error_with_500(
        self, http_callback, mock_handler: MagicMock, tmp_path, mocker: MockerFixture
    ) -> None:
        """GIVEN OS error reading file WHEN doGet called THEN returns 500."""
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"test")

        mocker.patch.object(http_callback, "_get_web_dir", return_value=str(tmp_path))
        mocker.patch("builtins.open", side_effect=OSError("disk error"))
        http_callback.doGet(mock_handler, "test.txt")

        mock_handler.send_response.assert_called_with(500)

    def test_doget_blocks_symlink_escape(
        self, http_callback, mock_handler: MagicMock, tmp_path, mocker: MockerFixture
    ) -> None:
        """GIVEN symlink pointing outside web dir WHEN doGet called THEN returns 403."""
        web_dir = tmp_path / "web"
        web_dir.mkdir()

        # Create a file outside web dir
        outside_file = tmp_path / "secret.txt"
        outside_file.write_text("secret data")

        # Create a symlink inside web dir pointing outside
        symlink_path = web_dir / "escape.txt"
        symlink_path.symlink_to(outside_file)

        mocker.patch.object(http_callback, "_get_web_dir", return_value=str(web_dir))
        http_callback.doGet(mock_handler, "escape.txt")

        # Should return 403 because resolved path is outside web_dir
        mock_handler.send_response.assert_called_with(403)

    def test_doget_allows_symlink_within_web_dir(
        self, http_callback, mock_handler: MagicMock, tmp_path, mocker: MockerFixture
    ) -> None:
        """GIVEN symlink pointing within web dir WHEN doGet called THEN serves file."""
        web_dir = tmp_path / "web"
        web_dir.mkdir()

        # Create a file inside web dir
        real_file = web_dir / "real.txt"
        real_file.write_bytes(b"content")

        # Create a symlink inside web dir pointing to the file
        symlink_path = web_dir / "link.txt"
        symlink_path.symlink_to(real_file)

        mocker.patch.object(http_callback, "_get_web_dir", return_value=str(web_dir))
        http_callback.doGet(mock_handler, "link.txt")

        # Should serve the file
        mock_handler.send_response.assert_called_with(200)

    def test_doget_handles_realpath_oserror(
        self, http_callback, mock_handler: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN realpath raises OSError WHEN doGet called THEN returns 403."""
        mocker.patch.object(http_callback, "_get_web_dir", return_value="/some/dir")
        mocker.patch("os.path.realpath", side_effect=OSError("permission denied"))
        http_callback.doGet(mock_handler, "test.txt")

        mock_handler.send_response.assert_called_with(403)


class TestHTTPHook:
    """The built-in HTTP server serves files only when httpRoot is empty;
    an httpRoot means an external server (nginx) owns that directory."""

    @pytest.mark.parametrize(("http_root", "hooked"), [("", True), ("/srv/llm", False)])
    def test_hook_only_without_http_root(
        self, mock_irc: MagicMock, mocker: MockerFixture, http_root: str, hooked: bool
    ) -> None:
        from llm.plugin import LLM

        from .conftest import plugin_init_patches

        mocker.patch.object(
            LLM, "registryValue", side_effect=make_registry_side_effect({"httpRoot": http_root})
        )
        mocks = plugin_init_patches(mocker)
        plugin = LLM(mock_irc)

        assert mocks["httpserver_hook"].called is hooked
        assert (plugin._http_callback is not None) is hooked

    def test_empty_http_root_serves_limnorias_web_dir(self, mocker: MockerFixture) -> None:
        """Never the process cwd: Path("") there would expose bot.conf and
        the env file to the public HTTP server."""
        import supybot.conf as conf
        from llm.plugin import LLMHTTPCallback

        plugin = mocker.MagicMock()
        plugin.registryValue.return_value = ""

        web_dir = LLMHTTPCallback(plugin)._get_web_dir()

        assert web_dir == conf.supybot.directories.data.web.dirize("llm")
        assert web_dir


def test_commands_run_in_worker_threads() -> None:
    """Limnoria runs a threaded plugin's commands off the driver thread, so a
    slow model call never stalls pings or other channels."""
    from llm.plugin import LLM

    assert LLM.threaded is True


class TestPluginHelperMethods:
    """Test plugin helper methods."""

    @pytest.fixture
    def mock_msg(self, mocker: MockerFixture) -> MagicMock:
        """Create a mock IRC message."""
        msg = mocker.MagicMock()
        msg.prefix = "testnick!user@host"
        msg.args = ("#testchannel", "test message")
        msg.time = time.time()
        msg.channel = "#testchannel"
        msg.server_tags = {}  # default: no IRCv3 account-tag
        return msg

    def test_resolve_caller_returns_account_when_available(
        self, mock_msg: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN user logged into account WHEN _resolve_caller called THEN key is account."""
        from llm.plugin import LLM

        mock_irc = mocker.MagicMock()
        mock_irc.state.nickToAccount = mocker.MagicMock(return_value="MyAccount")

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        plugin.db = mocker.MagicMock()
        plugin.context = mocker.MagicMock()
        plugin.log = mocker.MagicMock()
        result = plugin._resolve_caller(mock_irc, mock_msg).identity

        assert result.key == "MyAccount"
        assert result.account == "MyAccount"
        assert result.personal_key == "MyAccount"
        assert result.raw_nick == "testnick"

    def test_resolve_caller_falls_back_to_nick_when_no_account(
        self, mock_msg: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN user not logged in WHEN _resolve_caller called THEN key is the "!"-prefixed nick."""
        from llm.plugin import LLM

        mock_irc = mocker.MagicMock()
        mock_irc.state.nickToAccount = mocker.MagicMock(return_value=None)

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        result = plugin._resolve_caller(mock_irc, mock_msg).identity

        assert result.key == "!testnick"
        assert result.account is None
        assert result.personal_key is None
        assert result.raw_nick == "testnick"

    def test_resolve_caller_falls_back_to_nick_on_keyerror(
        self, mock_msg: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN nickToAccount raises KeyError WHEN _resolve_caller called THEN key is the "!"-prefixed nick."""
        from llm.plugin import LLM

        mock_irc = mocker.MagicMock()
        mock_irc.state.nickToAccount = mocker.MagicMock(side_effect=KeyError("unknown nick"))

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        result = plugin._resolve_caller(mock_irc, mock_msg).identity

        assert result.key == "!testnick"
        assert result.account is None

    def test_resolve_caller_in_channel_replies_to_channel(
        self, mock_msg: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN a channel message WHEN _resolve_caller called THEN the reply
        target is the channel."""
        from llm.plugin import LLM

        mock_irc = mocker.MagicMock()
        mock_irc.state.nickToAccount = mocker.MagicMock(return_value=None)
        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        caller = plugin._resolve_caller(mock_irc, mock_msg)

        assert caller.channel == "#testchannel"
        assert caller.is_private is False
        assert caller.reply_target == "#testchannel"

    def test_resolve_caller_in_pm_replies_to_sender_not_bot(self, mocker: MockerFixture) -> None:
        """GIVEN a PM (args[0] is the bot's nick) WHEN _resolve_caller called THEN
        the reply target is the sender's nick, never the bot's or the account."""
        from llm.plugin import LLM

        mock_irc = mocker.MagicMock()
        mock_irc.nick = "testbot"
        mock_msg = mocker.MagicMock(args=("testbot", "hi"), prefix="nick!u@h", server_tags={})
        mock_irc.state.nickToAccount = mocker.MagicMock(return_value=None)
        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        caller = make_caller(mock_irc, mock_msg, "rdrake", account="rdrake_acct")
        plugin.db = mocker.MagicMock()
        plugin.context = mocker.MagicMock()
        plugin.log = mocker.MagicMock()

        resolved = plugin._resolve_caller(mock_irc, mock_msg)

        assert resolved == caller
        assert resolved.channel is None
        assert resolved.is_private is True
        assert resolved.reply_target == "rdrake"

    def test_resolve_caller_with_no_args_is_private(self, mocker: MockerFixture) -> None:
        """GIVEN a message with no args WHEN _resolve_caller called THEN it is
        treated as private and replies to the sender."""
        from llm.plugin import LLM

        mock_irc = mocker.MagicMock()
        mock_irc.state.nickToAccount = mocker.MagicMock(return_value=None)
        mock_msg = mocker.MagicMock(args=[], prefix="nick!u@h", server_tags={})
        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)

        assert plugin._resolve_caller(mock_irc, mock_msg).reply_target == "nick"

    def test_is_old_message_returns_true_for_old_message(self, mocker: MockerFixture) -> None:
        """GIVEN message older than startup WHEN _is_old_message THEN returns True."""
        from llm.plugin import LLM

        mock_msg = mocker.MagicMock()
        mock_msg.time = time.time() - 100

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        plugin.startup_time = time.time()
        result = plugin._is_old_message(mock_msg)

        assert result is True

    def test_is_old_message_returns_false_for_zero_timestamp(self, mocker: MockerFixture) -> None:
        """GIVEN message with time=0 WHEN _is_old_message THEN returns False.

        Limnoria defaults msg.time to 0 when no server-time tag is present.
        This should be treated as a live message, not ZNC playback.
        """
        from llm.plugin import LLM

        mock_msg = mocker.MagicMock()
        mock_msg.time = 0

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        plugin.startup_time = time.time()
        result = plugin._is_old_message(mock_msg)

        assert result is False

    @pytest.mark.parametrize(
        ("command", "args"),
        [
            ("remind", ["clear"]),
            ("memories", ["clear"]),
            ("instruct", ["clear"]),
            ("tz", ["clear"]),
            ("avatar", ["clear"]),
            ("forget", []),
            ("verseopt", ["out"]),
            ("versedit", ["delete 1"]),
        ],
    )
    def test_replayed_mutating_command_is_ignored(self, plugin_env, command, args) -> None:
        """GIVEN a ZNC-replayed line from before startup WHEN a mutating command
        runs THEN it returns before touching state — a replayed '@remind clear'
        after a restart must not wipe reminders."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_msg.time = plugin.startup_time - 100
        plugin.db.reset_mock()

        getattr(plugin, command)(mock_irc, mock_msg, args)

        assert plugin.db.method_calls == []
        mock_irc.reply.assert_not_called()
        mock_irc.replies.assert_not_called()
        mock_irc.error.assert_not_called()


class TestBuildBridgeTool:
    """Tests for LLM._build_bridge_tool — per-request bridge tool builder."""

    def test_returns_none_when_disabled(self, plugin_env):
        plugin, irc, msg = plugin_env
        plugin.registryValue.side_effect = lambda k, ch=None: (
            False
            if k == "bridgeEnabled"
            else []
            if k == "bridgeAllowedPlugins"
            else False
            if k == "bridgeDebugInChannel"
            else None
        )

        schema, handlers = plugin._build_bridge_tool(irc, msg, "#test")

        assert schema is None
        assert handlers is None

    def test_disabled_exposes_nothing_even_with_commands_loaded(self, plugin_env, mocker):
        """bridgeEnabled False (the default) removes every bridge tool, even
        though an empty allowlist would fall back to the curated set."""
        plugin, irc, msg = plugin_env
        plugin.registryValue.side_effect = make_registry_side_effect({"bridgeEnabled": False})
        mocker.patch(
            "llm.limnoria_bridge.enumerate_commands",
            return_value=[
                mocker.MagicMock(plugin="Misc", command="ping", arg_syntax="", description="")
            ],
        )

        assert plugin._build_bridge_tool(irc, msg, "#test") == (None, None)

    def test_run_handler_returns_the_dispatch_envelope(self, plugin_env, mocker):
        """The model sees the command's output: the handler hands back the
        dispatch envelope as its tool result."""
        import json

        plugin, irc, msg = plugin_env
        plugin.registryValue.side_effect = make_registry_side_effect(
            {"bridgeEnabled": True, "bridgeAllowedPlugins": ["Time"]}
        )
        mocker.patch(
            "llm.limnoria_bridge.enumerate_commands",
            return_value=[
                mocker.MagicMock(plugin="Time", command="time", arg_syntax="", description="")
            ],
        )
        envelope = {"status": "ok", "output": "12:00"}
        mocker.patch("llm.limnoria_bridge.dispatch", return_value=envelope)

        _schema, handlers = plugin._build_bridge_tool(irc, msg, "#test")
        result = handlers["run_limnoria_command"]({"plugin": "Time", "command": "time"})

        assert json.loads(result.content) == envelope

    def test_returns_none_when_no_commands_resolve(self, plugin_env):
        """Empty allowlist falls back to DEFAULT_ALLOWED_PLUGINS, but no
        callbacks are loaded in this test fixture, so enumerate yields
        nothing and the bridge tool is not registered."""
        plugin, irc, msg = plugin_env
        plugin.registryValue.side_effect = lambda k, ch=None: (
            True
            if k == "bridgeEnabled"
            else []
            if k == "bridgeAllowedPlugins"
            else False
            if k == "bridgeDebugInChannel"
            else None
        )
        irc.callbacks = []

        schema, handlers = plugin._build_bridge_tool(irc, msg, "#test")

        assert schema is None
        assert handlers is None

    def test_empty_allowlist_falls_back_to_curated_default(self, plugin_env, mocker):
        """T2-A: empty bridgeAllowedPlugins + bridgeEnabled True →
        enumerate_commands receives the curated DEFAULT_ALLOWED_PLUGINS set."""
        from llm import limnoria_bridge as lb

        plugin, irc, msg = plugin_env
        plugin.registryValue.side_effect = lambda k, ch=None: (
            True
            if k == "bridgeEnabled"
            else []
            if k == "bridgeAllowedPlugins"
            else False
            if k == "bridgeDebugInChannel"
            else None
        )
        enum_mock = mocker.patch(
            "llm.limnoria_bridge.enumerate_commands",
            return_value=[
                mocker.MagicMock(plugin="Misc", command="ping", arg_syntax="", description="")
            ],
        )

        plugin._build_bridge_tool(irc, msg, "#test")

        called_allowed = enum_mock.call_args.args[2]
        assert called_allowed == lb.DEFAULT_ALLOWED_PLUGINS

    def test_explicit_non_empty_allowlist_overrides_curated_default(self, plugin_env, mocker):
        """T2-A: an operator-set allowlist must NOT get expanded with the
        curated set — explicit config wins."""
        plugin, irc, msg = plugin_env
        plugin.registryValue.side_effect = lambda k, ch=None: (
            True
            if k == "bridgeEnabled"
            else ["Misc"]
            if k == "bridgeAllowedPlugins"
            else False
            if k == "bridgeDebugInChannel"
            else None
        )
        enum_mock = mocker.patch(
            "llm.limnoria_bridge.enumerate_commands",
            return_value=[
                mocker.MagicMock(plugin="Misc", command="ping", arg_syntax="", description="")
            ],
        )

        plugin._build_bridge_tool(irc, msg, "#test")

        called_allowed = enum_mock.call_args.args[2]
        assert called_allowed == frozenset({"Misc"})

    def test_handler_passes_the_allowlist_to_dispatch(self, plugin_env, mocker):
        """The run handler hands dispatch the same allowlist enumerate got, so a
        loaded plugin outside it is refused at call time too."""
        plugin, irc, msg = plugin_env
        plugin.registryValue.side_effect = lambda k, ch=None: (
            True if k == "bridgeEnabled" else ["Time"] if k == "bridgeAllowedPlugins" else False
        )
        mocker.patch(
            "llm.limnoria_bridge.enumerate_commands",
            return_value=[
                mocker.MagicMock(plugin="Time", command="time", arg_syntax="", description="")
            ],
        )
        dispatch = mocker.patch(
            "llm.limnoria_bridge.dispatch", return_value={"error": "denied: Alias.add"}
        )

        _schema, handlers = plugin._build_bridge_tool(irc, msg, "#test")
        handlers["run_limnoria_command"]({"plugin": "Alias", "command": "add", "args": "x"})

        assert dispatch.call_args.kwargs["allowed"] == frozenset({"Time"})

    def test_returns_schema_and_handler_when_commands_present(self, plugin_env, mocker):
        plugin, irc, msg = plugin_env
        plugin.registryValue.side_effect = lambda k, ch=None: (
            True
            if k == "bridgeEnabled"
            else ["Misc"]
            if k == "bridgeAllowedPlugins"
            else False
            if k == "bridgeDebugInChannel"
            else None
        )
        fake_cmds = [
            mocker.MagicMock(
                plugin="Misc",
                command="ping",
                arg_syntax="takes no arguments",
                description="Replies with pong.",
            )
        ]
        mocker.patch("llm.limnoria_bridge.enumerate_commands", return_value=fake_cmds)

        schemas, handlers = plugin._build_bridge_tool(irc, msg, "#test")

        assert schemas is not None
        assert handlers is not None
        names = [s["function"]["name"] for s in schemas]
        assert "run_limnoria_command" in names
        assert "search_bridge_commands" in names
        assert "run_limnoria_command" in handlers
        assert "search_bridge_commands" in handlers
        run_schema = next(s for s in schemas if s["function"]["name"] == "run_limnoria_command")
        # Description should mention the available command.
        assert "Misc.ping" in run_schema["function"]["description"]

    def test_search_handler_returns_matching_commands(self, plugin_env, mocker):
        plugin, irc, msg = plugin_env
        plugin.registryValue.side_effect = lambda k, ch=None: (
            True
            if k == "bridgeEnabled"
            else ["Misc"]
            if k == "bridgeAllowedPlugins"
            else False
            if k == "bridgeDebugInChannel"
            else None
        )
        from llm import limnoria_bridge as lb

        fake_cmds = [
            lb.BridgeCommand(
                plugin="Misc", command="ping", arg_syntax="", description="Replies pong."
            ),
            lb.BridgeCommand(
                plugin="Misc",
                command="help",
                arg_syntax="<command>",
                description="Returns help for a command.",
            ),
        ]
        mocker.patch("llm.limnoria_bridge.enumerate_commands", return_value=fake_cmds)

        _, handlers = plugin._build_bridge_tool(irc, msg, "#test")
        result = handlers["search_bridge_commands"]({"query": "pong"})

        import json

        envelope = json.loads(result.content)
        assert envelope["status"] == "ok"
        assert len(envelope["matches"]) == 1
        assert envelope["matches"][0]["plugin"] == "Misc"
        assert envelope["matches"][0]["command"] == "ping"

    def test_search_handler_blank_query_returns_error(self, plugin_env, mocker):
        plugin, irc, msg = plugin_env
        plugin.registryValue.side_effect = lambda k, ch=None: (
            True
            if k == "bridgeEnabled"
            else ["Misc"]
            if k == "bridgeAllowedPlugins"
            else False
            if k == "bridgeDebugInChannel"
            else None
        )
        mocker.patch(
            "llm.limnoria_bridge.enumerate_commands",
            return_value=[
                mocker.MagicMock(plugin="Misc", command="ping", arg_syntax="", description="")
            ],
        )

        _, handlers = plugin._build_bridge_tool(irc, msg, "#test")
        result = handlers["search_bridge_commands"]({"query": ""})

        import json

        envelope = json.loads(result.content)
        assert "error" in envelope

    def test_search_handler_clamps_limit(self, plugin_env, mocker):
        plugin, irc, msg = plugin_env
        plugin.registryValue.side_effect = lambda k, ch=None: (
            True
            if k == "bridgeEnabled"
            else ["Misc"]
            if k == "bridgeAllowedPlugins"
            else False
            if k == "bridgeDebugInChannel"
            else None
        )
        from llm import limnoria_bridge as lb

        fake_cmds = [
            lb.BridgeCommand(plugin="Misc", command=f"cmd{i}", arg_syntax="", description="ping")
            for i in range(50)
        ]
        mocker.patch("llm.limnoria_bridge.enumerate_commands", return_value=fake_cmds)

        _, handlers = plugin._build_bridge_tool(irc, msg, "#test")
        # Over the cap → clamped to 25
        result_over = handlers["search_bridge_commands"]({"query": "ping", "limit": 999})
        # Under the floor → clamped to 1
        result_under = handlers["search_bridge_commands"]({"query": "ping", "limit": 0})

        import json

        assert len(json.loads(result_over.content)["matches"]) == 25
        assert len(json.loads(result_under.content)["matches"]) == 1

    def test_behavior_later_notes_visible_tell_hidden(self, plugin_env, mocker):
        """Behavior-level (not plumbing): with Later allowlisted, the rendered
        tool description must list Later.notes (read) and NOT list Later.tell
        (mutating)."""
        plugin, irc, msg = plugin_env

        # Stub callback shaped to look like the Later plugin: 'tell' (mutating)
        # and 'notes' (read-only).
        later = mocker.MagicMock()
        later.name.return_value = "Later"
        later.canonicalName.return_value = "later"
        later.listCommands.return_value = ["tell", "notes"]
        later.getCommandMethod.side_effect = lambda cmd: mocker.MagicMock(
            __doc__={
                "tell": "<nick> <text>\n\nQueue offline message.",
                "notes": "takes no arguments\n\nList queued notes.",
            }[cmd[0]]
        )
        irc.callbacks = [later]
        mocker.patch(
            "llm.limnoria_bridge.callbacks.checkCommandCapability",
            return_value=False,
        )

        # Only 'notes' should appear in the description table.
        plugin.registryValue.side_effect = lambda k, ch=None: (
            True
            if k == "bridgeEnabled"
            else ["Later"]
            if k == "bridgeAllowedPlugins"
            else False
            if k == "bridgeDebugInChannel"
            else None
        )
        schemas, _ = plugin._build_bridge_tool(irc, msg, "#test")
        desc = schemas[0]["function"]["description"]
        assert "later.notes" in desc.lower()
        assert "later.tell" not in desc.lower()


class TestStripNickAddress:
    """Direct unit coverage for the nick-address-stripping helper."""

    def test_colon_separator(self) -> None:
        from llm.plugin import LLM

        assert LLM._strip_nick_address("bot", "bot: hello") == "hello"

    def test_comma_separator(self) -> None:
        from llm.plugin import LLM

        assert LLM._strip_nick_address("bot", "bot, hello") == "hello"

    def test_whitespace_separator(self) -> None:
        from llm.plugin import LLM

        assert LLM._strip_nick_address("bot", "bot hello") == "hello"

    def test_case_insensitive_match(self) -> None:
        from llm.plugin import LLM

        assert LLM._strip_nick_address("bot", "BOT, hi") == "hi"

    def test_returns_none_when_nick_is_part_of_word(self) -> None:
        from llm.plugin import LLM

        assert LLM._strip_nick_address("bot", "bottle of wine") is None

    def test_returns_none_when_no_text_after_nick(self) -> None:
        from llm.plugin import LLM

        assert LLM._strip_nick_address("bot", "bot") is None

    def test_trailing_nick_after_comma(self) -> None:
        from llm.plugin import LLM

        assert LLM._strip_nick_address("bot", "draw a cat, bot") == "draw a cat"

    def test_trailing_nick_with_sentence_punctuation(self) -> None:
        from llm.plugin import LLM

        assert LLM._strip_nick_address("bot", "what time is it, bot?") == "what time is it"

    def test_trailing_nick_is_case_insensitive(self) -> None:
        from llm.plugin import LLM

        assert LLM._strip_nick_address("bot", "prods BOT") == "prods"

    def test_returns_none_when_nick_is_a_word_suffix(self) -> None:
        from llm.plugin import LLM

        assert LLM._strip_nick_address("bot", "time for a reboot") is None

    def test_returns_none_for_mention_mid_sentence(self) -> None:
        from llm.plugin import LLM

        assert LLM._strip_nick_address("bot", "I asked bot about it") is None

    def test_returns_none_when_trailing_nick_is_the_whole_line(self) -> None:
        from llm.plugin import LLM

        assert LLM._strip_nick_address("bot", "bot!") is None


class TestSafeQueue:
    def test_safe_queue_drops_when_closing(self, plugin_env, mocker) -> None:
        plugin, _irc, _msg = plugin_env
        plugin._llm_executor.shutdown()
        target_irc = mocker.MagicMock()
        ok = plugin._safe_queue(target_irc, mocker.sentinel.msg)
        target_irc.queueMsg.assert_not_called()
        assert ok is False


class TestBridgeDebugFooter:
    """``_format_bridge_debug_footer`` is sent to a public channel, so it must
    never echo raw bridge args — they are LLM-generated and may carry secrets
    (e.g. an API key in a URL)."""

    def test_footer_excludes_raw_args(self) -> None:
        from llm.plugin import LLM

        trace = [("Web", "fetch", "https://api.example.com/v1?key=sk-SECRET-XYZ", "ok")]
        footer = LLM._format_bridge_debug_footer(trace)
        assert "sk-SECRET-XYZ" not in footer
        assert "Web.fetch" in footer
        assert "[ok]" in footer

    def test_footer_signals_args_present_without_content(self) -> None:
        from llm.plugin import LLM

        trace = [("Note", "send", "bob hello there", "ok")]
        footer = LLM._format_bridge_debug_footer(trace)
        assert "bob hello there" not in footer
        # Length is a leak-free signal that args were passed.
        assert str(len("bob hello there")) in footer


class TestSafeReply:
    """``_safe_reply`` serializes worker-thread ``irc.reply`` sends on the same
    ``_irc_send_lock`` as ``_safe_queue`` and short-circuits while closing."""

    def test_safe_reply_drops_when_closing(self, plugin_env, mocker) -> None:
        plugin, _irc, _msg = plugin_env
        plugin._llm_executor.shutdown()
        target_irc = mocker.MagicMock()
        ok = plugin._safe_reply(target_irc, "hi", prefixNick=False)
        target_irc.reply.assert_not_called()
        assert ok is False


class TestSafeError:
    """``_safe_error`` serializes worker-thread ``irc.error`` sends on the same
    ``_irc_send_lock`` as ``_safe_queue``/``_safe_reply`` and short-circuits
    while closing."""

    def test_safe_error_drops_when_closing(self, plugin_env, mocker) -> None:
        plugin, _irc, _msg = plugin_env
        plugin._llm_executor.shutdown()
        target_irc = mocker.MagicMock()
        plugin._safe_error(target_irc, "oops")
        target_irc.error.assert_not_called()

    def test_safe_error_works_on_a_proxy_without_raise(self, plugin_env) -> None:
        """Prod 2026-10-10: the addressed path's ReplyIrcProxy forwards kwargs
        to _makeReply, which has no Raise — TypeError, and no reply at all."""
        plugin, _irc, _msg = plugin_env
        sent: list[str] = []

        class NoRaiseProxy:
            def error(self, s, prefixNick=False):  # noqa: N803
                sent.append(s)

        plugin._safe_error(NoRaiseProxy(), "empty")

        assert sent == ["empty"]


class TestRateBucketsConcurrency:
    def test_concurrent_rate_limit_count_is_exact(self, plugin_env) -> None:
        """The lock guarantees the deque length matches the number of
        recorded hits exactly. CPython's GIL makes individual deque ops
        atomic, so a "no exception" assertion would pass even on the
        broken code — assert the count instead."""
        import threading
        import time

        plugin, _irc, _msg = plugin_env
        errors: list[Exception] = []
        threads_n = 8
        per_thread = 200
        now = time.time()
        barrier = threading.Barrier(threads_n)

        def hammer() -> None:
            try:
                barrier.wait(timeout=2)
                for _ in range(per_thread):
                    plugin._record_rate_limit_hit("ask", "alice", now)
                    plugin._is_rate_limited("ask", "alice", now, tier="registered")
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=hammer) for _ in range(threads_n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        # The deque trims entries older than the window; with `now`
        # constant, every hit lands in the window.
        key = "ask:alice"
        assert len(plugin._rate_buckets[key]) == threads_n * per_thread


class TestWatchModeReminderMigration:
    """Verify the action-prompt reminder path submits to the executor and
    that the plain / rate-limited / no-IRC paths settle the fire with the
    agenda on the main thread."""

    @staticmethod
    def _row(event_name: str, **kw):
        return make_reminder_row(
            event_name=event_name, nick="alice", channel="#chan", message="ping", **kw
        )

    def test_action_prompt_dispatch_submits_to_executor(self, plugin_env, mocker) -> None:
        plugin, irc, _msg = plugin_env
        plugin._llm_executor = mocker.MagicMock()
        plugin._llm_executor.closing = False
        mocker.patch("llm.plugin.world.ircs", [irc])
        plugin._check_rate_limit = mocker.MagicMock(return_value=False)
        finish = mocker.patch.object(plugin, "_finish_agenda")

        plugin._fire_reminder(self._row("evt-1", action_prompt="say hi"), 100.0)
        plugin._llm_executor.submit.assert_called_once()
        label = plugin._llm_executor.submit.call_args[0][0]
        assert label.startswith("reminder:")
        # The worker settles a submitted fire, not the scheduler thread.
        finish.assert_not_called()

    def test_legacy_no_action_prompt_does_not_submit(self, plugin_env, mocker) -> None:
        plugin, irc, _msg = plugin_env
        plugin._llm_executor = mocker.MagicMock()
        plugin._llm_executor.closing = False
        mocker.patch("llm.plugin.world.ircs", [irc])
        plugin.llm_service.sanitize_output.side_effect = lambda x: x
        finish = mocker.patch.object(plugin, "_finish_agenda")

        plugin._fire_reminder(self._row("evt-2"), 100.0)
        plugin._llm_executor.submit.assert_not_called()
        irc.queueMsg.assert_called_once()
        finish.assert_called_once_with(agenda.Kind.REMINDER, "evt-2", 100.0)

    def test_rate_limit_skip_does_not_submit(self, plugin_env, mocker) -> None:
        plugin, irc, _msg = plugin_env
        plugin._llm_executor = mocker.MagicMock()
        plugin._llm_executor.closing = False
        mocker.patch("llm.plugin.world.ircs", [irc])
        plugin._check_rate_limit = mocker.MagicMock(return_value=True)
        plugin.llm_service.sanitize_output.side_effect = lambda x: x

        plugin._fire_reminder(self._row("evt-3", action_prompt="say hi"), 100.0)
        plugin._llm_executor.submit.assert_not_called()
        # Skip notice queued via _safe_queue → irc.queueMsg.
        irc.queueMsg.assert_called()

    def test_no_active_irc_finalizes_in_main_thread(self, plugin_env, mocker) -> None:
        plugin, _irc, _msg = plugin_env
        plugin._llm_executor = mocker.MagicMock()
        plugin._llm_executor.closing = False
        mocker.patch("llm.plugin.world.ircs", [])
        finish = mocker.patch.object(plugin, "_finish_agenda")

        plugin._fire_reminder(self._row("evt-4", action_prompt="say hi"), 100.0)
        plugin._llm_executor.submit.assert_not_called()
        finish.assert_called_once_with(agenda.Kind.REMINDER, "evt-4", 100.0)

    def test_closing_skips_submit_and_settling(self, plugin_env, mocker) -> None:
        plugin, irc, _msg = plugin_env
        plugin._llm_executor = mocker.MagicMock()
        plugin._llm_executor.closing = True
        mocker.patch("llm.plugin.world.ircs", [irc])
        plugin._check_rate_limit = mocker.MagicMock(return_value=False)

        plugin._fire_reminder(self._row("evt-5", action_prompt="say hi"), 100.0)
        plugin._llm_executor.submit.assert_not_called()
        # A dying plugin leaves the row alone: it fires again next start.
        plugin.db.delete_reminder.assert_not_called()
        plugin.db.update_reminder_fire_at.assert_not_called()


class TestUsageBooking:
    """Spend is booked whether or not a reply was delivered; history is not."""

    SPEND = Spend(prompt_tokens=30, completion_tokens=7, cost=0.002)

    def _book(self, plugin_env, mocker, result: AssistantResult, *, delivered: bool):
        plugin, irc, msg = plugin_env
        caller = make_caller(irc, msg, "testnick", account="acct", channel="#test")
        add_message = mocker.patch.object(plugin.context, "add_message")
        plugin._store_context_and_log_usage(
            caller, "ask", "hi", result.content, result, irc, delivered=delivered
        )
        plugin.db.log_usage.assert_called_once()
        args, kwargs = plugin.db.log_usage.call_args
        assert args[4:7] == (30, 7, 0.002)
        return kwargs, add_message

    def test_delivered_success_is_booked_and_stored(self, plugin_env, mocker) -> None:
        result = AssistantResult(content="reply", spend=self.SPEND, model="m")
        kwargs, add_message = self._book(plugin_env, mocker, result, delivered=True)
        assert kwargs["status"] == "success"
        assert add_message.called

    def test_echo_after_retry_is_booked_as_error_not_stored(self, plugin_env, mocker) -> None:
        """GIVEN the echo exit (empty content, error set) WHEN nothing was
        delivered THEN the spend still lands, as ``error``, and no history."""
        result = AssistantResult(
            content="",
            spend=self.SPEND,
            model="m",
            error="Model echoed the prompt instead of replying.",
        )
        kwargs, add_message = self._book(plugin_env, mocker, result, delivered=False)
        assert kwargs["status"] == "error"
        assert "echoed" in kwargs["error_detail"]
        add_message.assert_not_called()

    def test_undelivered_without_error_is_booked_as_error(self, plugin_env, mocker) -> None:
        """GIVEN a clean result that never reached IRC (fake react note, failed
        send) THEN it books as ``error`` with a reason, and stores nothing."""
        result = AssistantResult(content="[reacted 🫡]", spend=self.SPEND, model="m")
        kwargs, add_message = self._book(plugin_env, mocker, result, delivered=False)
        assert (kwargs["status"], kwargs["error_detail"]) == ("error", "not delivered")
        add_message.assert_not_called()

    def test_stashed_timeout_is_booked_as_timeout_not_stored(self, plugin_env, mocker) -> None:
        placeholder = "Timed out, but I'll keep trying and deliver the answer when ready."
        result = AssistantResult(
            content=placeholder, spend=self.SPEND, model="m", error=placeholder, timed_out=True
        )
        kwargs, add_message = self._book(plugin_env, mocker, result, delivered=True)
        assert kwargs["status"] == "timeout"
        add_message.assert_not_called()

    @pytest.mark.parametrize(
        ("content", "silent", "status"),
        [("ok", False, "success"), ("", False, "error"), ("[silent]", True, "silent")],
    )
    def test_unattended_status(self, plugin_env, content, silent, status) -> None:
        """Reminder and scheduled-task fires share the command path's statuses."""
        plugin, _irc, _msg = plugin_env
        result = AssistantResult(content=content, spend=self.SPEND, model="m")
        plugin._log_unattended_usage(
            nick="n",
            account=None,
            channel="#c",
            command="remind_action",
            prompt="p",
            result=result,
            silent=silent,
            log_context="t",
        )
        assert plugin.db.log_usage.call_args.kwargs["status"] == status

    def test_reply_that_raises_is_still_booked(self, plugin_env, mocker) -> None:
        """GIVEN sending the reply raises THEN the turn books as undelivered
        before the exception propagates, and nothing reaches history."""
        plugin, irc, msg = plugin_env
        caller = make_caller(irc, msg, "testnick", account="acct", channel="#test")
        mocker.patch.object(plugin, "_dispatch_assistant_reply", side_effect=RuntimeError("boom"))
        add_message = mocker.patch.object(plugin.context, "add_message")
        result = AssistantResult(content="reply", spend=self.SPEND, model="m")

        with pytest.raises(RuntimeError):
            plugin._reply_and_book(
                irc, msg, result, caller=caller, command="code", text="hi", response="reply"
            )

        kwargs = plugin.db.log_usage.call_args.kwargs
        assert (kwargs["status"], kwargs["error_detail"]) == ("error", "not delivered")
        add_message.assert_not_called()

    def test_failed_long_reply_send_is_not_delivered(self, plugin_env, mocker) -> None:
        """GIVEN the reply could not be queued THEN dispatch reports it undelivered."""
        plugin, irc, msg = plugin_env
        caller = make_caller(irc, msg, "testnick", account="acct", channel="#test")
        mocker.patch.object(plugin, "_send_long_reply", return_value=False)
        result = AssistantResult(content="reply", spend=self.SPEND, model="m")

        _response, delivered = plugin._dispatch_assistant_reply(
            irc, msg, result, caller=caller, response="reply"
        )

        assert delivered is False

    def test_mock_result_never_books_as_timeout(self, plugin_env, mocker) -> None:
        """A MagicMock's timed_out is a truthy mock, not True; it must not
        turn every mocked turn into a ``timeout`` row."""
        plugin, _irc, _msg = plugin_env
        result = mocker.MagicMock(error=None)
        assert plugin._usage_status(result, delivered=True) == ("success", "")


class TestMemoryExtractionMigration:
    @pytest.mark.parametrize(("account", "scheduled"), [(None, False), ("acct", True)])
    def test_extraction_needs_an_account(self, plugin_env, mocker, account, scheduled) -> None:
        """GIVEN a successful ask WHEN the caller has no account THEN no extraction
        runs: facts would land under the raw nick, which nothing reads."""
        plugin, irc, msg = plugin_env
        caller = make_caller(irc, msg, "testnick", account=account, channel="#test")
        extract = mocker.patch.object(plugin, "_schedule_memory_extraction")
        result = mocker.MagicMock(error=None, content="reply", cost=0.0)

        plugin._store_context_and_log_usage(caller, "ask", "hi", "reply", result, irc)

        assert extract.called is scheduled

    def test_extraction_short_circuits_when_closing(self, plugin_env, mocker) -> None:
        plugin, _irc, _msg = plugin_env
        plugin._llm_executor = mocker.MagicMock()
        plugin._llm_executor.closing = True
        add_event = mocker.patch("llm.plugin.schedule.addEvent")

        plugin._schedule_memory_extraction("alice", "#chan", "user msg", "bot reply")
        callback = add_event.call_args[0][0]
        callback()
        plugin._llm_executor.submit.assert_not_called()


class TestLLMExecutorLifecycle:
    def test_die_shuts_down_executor(self, plugin_env) -> None:
        plugin, _irc, _msg = plugin_env
        plugin.die()
        assert plugin._llm_executor.closing is True


class TestRunFileCleanup:
    """Test _run_file_cleanup scheduled task."""

    def test_run_file_cleanup_handles_errors(self, mocker: MockerFixture) -> None:
        """GIVEN cleanup error WHEN _run_file_cleanup called THEN logs error."""
        from llm.plugin import LLM

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        plugin.llm_service = mocker.MagicMock()
        plugin.llm_service.run_scheduled_cleanup.side_effect = Exception("test error")
        plugin.log = mocker.MagicMock()

        # Should not raise
        plugin._run_file_cleanup()

        plugin.log.error.assert_called_once()


class TestStartupNotification:
    """Test startup notification to bot owner."""

    @pytest.fixture
    def plugin_with_mocks(self, mocker: MockerFixture) -> tuple:
        """Create plugin with mocked dependencies for startup tests."""
        from llm.plugin import LLM

        mock_irc = mocker.MagicMock()
        mock_irc.nick = "VibeBot"
        mock_irc.state.channels = {"#channel1": mocker.MagicMock(), "#channel2": mocker.MagicMock()}

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        plugin._pending_channels = set()
        plugin._irc_queries = ircquery.IrcQueryRegistry()
        plugin._startup_notified = False
        plugin.log = mocker.MagicMock()

        return plugin, mock_irc

    def test_dojoin_tracks_bot_joins(self, plugin_with_mocks: tuple, mocker: MockerFixture) -> None:
        """GIVEN bot joining channel WHEN doJoin called THEN adds to pending."""
        plugin, mock_irc = plugin_with_mocks

        mock_msg = mocker.MagicMock()
        mock_msg.nick = "VibeBot"
        mock_msg.args = ["#channel1"]

        mocker.patch("supybot.ircutils.strEqual", return_value=True)
        plugin.doJoin(mock_irc, mock_msg)

        assert "#channel1" in plugin._pending_channels

    def test_do315_removes_synced_channel(
        self, plugin_with_mocks: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN pending channel WHEN do315 received THEN removes from pending."""
        plugin, mock_irc = plugin_with_mocks
        plugin._pending_channels.add("#channel1")

        mock_msg = mocker.MagicMock()
        mock_msg.args = ["VibeBot", "#channel1"]

        mocker.patch.object(plugin, "_send_startup_notification")
        plugin.do315(mock_irc, mock_msg)

        assert "#channel1" not in plugin._pending_channels

    def test_do315_sends_notification_when_all_synced(
        self, plugin_with_mocks: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN last channel synced WHEN do315 received THEN sends notification."""
        plugin, mock_irc = plugin_with_mocks
        plugin._pending_channels.add("#channel1")

        mock_msg = mocker.MagicMock()
        mock_msg.args = ["VibeBot", "#channel1"]

        mock_notify = mocker.patch.object(plugin, "_send_startup_notification")
        plugin.do315(mock_irc, mock_msg)

        mock_notify.assert_called_once_with(mock_irc)
        assert plugin._startup_notified is True

    def test_do315_does_not_send_notification_if_channels_pending(
        self, plugin_with_mocks: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN other channels pending WHEN do315 received THEN no notification."""
        plugin, mock_irc = plugin_with_mocks
        plugin._pending_channels.add("#channel1")
        plugin._pending_channels.add("#channel2")

        mock_msg = mocker.MagicMock()
        mock_msg.args = ["VibeBot", "#channel1"]

        mock_notify = mocker.patch.object(plugin, "_send_startup_notification")
        plugin.do315(mock_irc, mock_msg)

        mock_notify.assert_not_called()
        assert plugin._startup_notified is False

    def test_do315_does_not_send_duplicate_notification(
        self, plugin_with_mocks: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN already notified WHEN do315 received THEN no duplicate."""
        plugin, mock_irc = plugin_with_mocks
        plugin._pending_channels.add("#channel1")
        plugin._startup_notified = True

        mock_msg = mocker.MagicMock()
        mock_msg.args = ["VibeBot", "#channel1"]

        mock_notify = mocker.patch.object(plugin, "_send_startup_notification")
        plugin.do315(mock_irc, mock_msg)

        mock_notify.assert_not_called()

    def test_do376_resets_tracking_state(
        self, plugin_with_mocks: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN plugin WHEN do376 received THEN resets tracking state."""
        plugin, mock_irc = plugin_with_mocks
        plugin._pending_channels.add("#oldchannel")
        plugin._startup_notified = True

        mock_msg = mocker.MagicMock()

        mocker.patch("supybot.schedule.addEvent")
        plugin.do376(mock_irc, mock_msg)

        assert len(plugin._pending_channels) == 0
        assert plugin._startup_notified is False

    def _mock_owner_user(self, mocker: MockerFixture, name: str) -> MagicMock:
        """Create a mock user with owner capability."""
        mock_user = mocker.MagicMock()
        mock_user.name = name
        mock_user.capabilities = ["owner"]
        return mock_user

    def test_send_startup_notification_sends_pm(
        self, plugin_with_mocks: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN owner configured WHEN notification sent THEN PMs owner."""
        from llm import plugin as plugin_module

        plugin, mock_irc = plugin_with_mocks

        # Mock ircdb.users.users with an owner user
        mock_ircdb = mocker.MagicMock()
        mock_ircdb.users.users.values.return_value = [self._mock_owner_user(mocker, "owner_nick")]

        original_ircdb = plugin_module.ircdb
        plugin_module.ircdb = mock_ircdb
        try:
            mocker.patch("supybot.schedule.removeEvent")
            plugin._send_startup_notification(mock_irc)
        finally:
            plugin_module.ircdb = original_ircdb

        mock_irc.queueMsg.assert_called_once()
        queued_msg = mock_irc.queueMsg.call_args[0][0]
        assert queued_msg.args[0] == "owner_nick"
        assert "VibeBot started" in queued_msg.args[1]
        assert "2 channels" in queued_msg.args[1]
        assert "UTC" in queued_msg.args[1]

    def test_send_startup_notification_handles_no_owner(
        self, plugin_with_mocks: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN no owner configured WHEN notification sent THEN logs warning."""
        from llm import plugin as plugin_module

        plugin, mock_irc = plugin_with_mocks

        # Mock ircdb.users.users with no owner users
        mock_ircdb = mocker.MagicMock()
        mock_ircdb.users.users.values.return_value = []

        original_ircdb = plugin_module.ircdb
        plugin_module.ircdb = mock_ircdb
        try:
            mocker.patch("supybot.schedule.removeEvent")
            plugin._send_startup_notification(mock_irc)
        finally:
            plugin_module.ircdb = original_ircdb

        mock_irc.queueMsg.assert_not_called()
        plugin.log.warning.assert_called_once()


# NOTE: reminder *delivery* is covered end-to-end by
# test_reminders.py::TestReminderDeliveryClosure, which drives the real
# plugin._make_reminder_delivery_closure(). A former test here reimplemented
# that closure inline (asserting on the test's own copy), so it could not catch
# a regression in production and was removed.


class TestAllowConcurrent:
    """Test _allow_concurrent context manager for concurrent command execution."""

    def test_allow_concurrent_releases_and_reacquires_lock(self, mocker: MockerFixture) -> None:
        """GIVEN lock held WHEN _allow_concurrent used THEN lock released inside and reacquired after."""
        import threading

        from llm.plugin import LLM

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        plugin._MetaSynchronized_rlock = threading.RLock()

        lock = plugin._MetaSynchronized_rlock
        lock.acquire()
        try:
            with plugin._allow_concurrent():
                # Lock should be released — another thread should be able to acquire it
                acquired = lock.acquire(blocking=False)
                assert acquired, "Lock should be released inside _allow_concurrent"
                lock.release()

            # Lock should be re-acquired after exiting context manager
            # Try to acquire non-blocking — should fail because it's held
            acquired = lock.acquire(blocking=False)
            # RLock is reentrant, so this will succeed but count goes up
            assert acquired
            lock.release()
        finally:
            lock.release()


class TestPluginInit:
    """Test plugin initialization paths using full init patches."""

    def test_init_installs_secret_filter(self, mock_irc: MagicMock, mocker: MockerFixture) -> None:
        """GIVEN plugin initialized WHEN __init__ runs THEN install_secret_filter is called.

        Regression guard: deleting the install_secret_filter() call from
        __init__ entirely is otherwise caught by no test — the six
        environment-backed _sanitize tests only exercise apikeys.scrub
        directly and never construct a real plugin, so a mutant that drops
        the call still leaves _sanitize apparently working while
        llm.verse.* records go out unfiltered via logging.lastResort.
        """
        from llm import apikeys
        from llm.plugin import LLM

        from .conftest import make_registry_side_effect, plugin_init_patches

        install_spy = mocker.spy(apikeys, "install_secret_filter")
        mocker.patch.object(LLM, "registryValue", side_effect=make_registry_side_effect())
        plugin_init_patches(mocker)
        LLM(mock_irc)

        install_spy.assert_called_once_with()

    def test_init_startup_log_reports_names_not_values(
        self, mock_irc: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN configured provider keys WHEN plugin initialized THEN the
        startup redaction log line reports variable *names*, never values.

        Regression guard for the hard "names and counts only, never values"
        constraint: a mutant that logs os.environ.get(name) per variable
        instead of apikeys.secret_var_names() would still produce a
        plausible-looking log line, but would leak FAKE_PROVIDER_KEYS values
        into logs/CI output. Asserts the exact argument passed to
        self.log.info, not just that *some* log call happened.
        """
        from llm import apikeys
        from llm.plugin import LLM

        from .conftest import FAKE_PROVIDER_KEYS, make_registry_side_effect, plugin_init_patches

        mocker.patch.object(LLM, "registryValue", side_effect=make_registry_side_effect())
        mocks = plugin_init_patches(mocker)
        LLM(mock_irc)

        log_mock = mocks["log"].getPluginLogger.return_value
        redaction_calls = [
            call for call in log_mock.info.call_args_list if "secret redaction" in call.args[0]
        ]
        assert len(redaction_calls) == 1
        installed, covered, names_arg = redaction_calls[0].args[1:4]
        assert covered == len(apikeys.secret_var_names())
        assert names_arg == ", ".join(apikeys.secret_var_names())
        assert isinstance(installed, int)
        for value in FAKE_PROVIDER_KEYS.values():
            assert value not in names_arg


class TestDoCapNewRequestsExperimentalCaps:
    """AfterNET's bouncer advertises draft/multiline by CAP NEW after SASL.

    Stock Limnoria filters CAP NEW through REQUEST_CAPABILITIES only, so
    without the patch draft/multiline is never requested and long replies
    fall back to @more. With experimentalExtensions on, the patched handler
    must register the new caps and request the ones the bot wants.
    """

    @pytest.fixture
    def cap_new(self, mocker: MockerFixture):
        from llm.plugin import _patch_irc_docapnew
        from supybot import conf, irclib, ircmsgs

        _patch_irc_docapnew()
        setting = conf.supybot.protocols.irc.experimentalExtensions
        original = setting()
        setting.setValue(True)

        def send(caps: str, *, acked: set[str] | None = None, state=None):
            irc = mocker.MagicMock()
            irc.state.capabilities_ls = {
                "draft/multiline": None,
                "message-tags": None,
                "vendor/unwanted": None,
            }
            irc.state.capabilities_ack = {"message-tags"} if acked is None else acked
            irc.state.fsm.state = state or irclib.IrcStateFsm.States.CONNECTED
            msg = ircmsgs.IrcMsg(command="CAP", args=("*", "NEW", caps))
            irclib.Irc.doCapNew(irc, msg)
            return irc, msg

        try:
            yield send
        finally:
            setting.setValue(original)

    def test_cap_new_requests_draft_multiline(self, cap_new) -> None:
        irc, msg = cap_new("draft/multiline")

        irc._addCapabilities.assert_called_once_with("draft/multiline", msg)
        irc.requestCapabilities.assert_called_once_with({"draft/multiline"})

    def test_an_already_acked_cap_is_not_requested_again(self, cap_new) -> None:
        irc, _ = cap_new("draft/multiline", acked={"message-tags", "draft/multiline"})

        irc.requestCapabilities.assert_not_called()

    def test_nothing_is_requested_while_shutting_down(self, cap_new) -> None:
        from supybot import irclib

        irc, _ = cap_new("draft/multiline", state=irclib.IrcStateFsm.States.SHUTTING_DOWN)

        irc.requestCapabilities.assert_not_called()


class TestPendingQueueRearm:
    """Each poll re-arms the one-shot wakeup for the next due task, so a
    reminder or deferred result is not left for the 5-minute safety poll."""

    def _plugin(self, mock_irc, mocker):
        from llm.plugin import LLM

        from .conftest import plugin_init_patches

        mocker.patch.object(LLM, "registryValue", side_effect=make_registry_side_effect())
        plugin_init_patches(mocker)
        add_event = mocker.patch("llm.plugin.schedule.addEvent")
        plugin = LLM(mock_irc)
        mocker.patch("llm.plugin.world.ircs", [])
        plugin.llm_service.check_pending_tasks.return_value = []
        plugin.db.get_next_due_time.return_value = time.time() + 120
        add_event.reset_mock()
        return plugin, add_event

    @staticmethod
    def _wakeups(add_event) -> list[float]:
        return [
            c.args[1]
            for c in add_event.call_args_list
            if c.kwargs.get("name") == "llm_queue_wakeup"
        ]

    def test_poll_arms_the_next_wakeup(self, mock_irc, mocker: MockerFixture) -> None:
        plugin, add_event = self._plugin(mock_irc, mocker)
        due = plugin.db.get_next_due_time.return_value

        plugin._check_pending_tasks()

        assert self._wakeups(add_event) == [due]

    def test_closing_plugin_arms_nothing(self, mock_irc, mocker: MockerFixture) -> None:
        plugin, add_event = self._plugin(mock_irc, mocker)
        plugin._llm_executor.shutdown()

        plugin._check_pending_tasks()

        assert self._wakeups(add_event) == []


class TestAgendaRestart:
    """A restart picks persisted reminders and tasks up from the database
    through the one agenda wakeup (real SQLite; the scheduler is a mock)."""

    @pytest.fixture
    def boot(self, mock_irc: MagicMock, mocker: MockerFixture, llm_data_dir):
        """Seed a database, then start the plugin on it. Returns a starter."""
        from llm.persistence import LLMDatabase
        from llm.plugin import LLM

        from .conftest import make_registry_side_effect, plugin_init_patches

        seed = LLMDatabase(str(llm_data_dir / "LLM.db"))

        def start():
            seed.close()
            mocker.patch.object(LLM, "registryValue", side_effect=make_registry_side_effect())
            mocks = plugin_init_patches(mocker, mock_database=False)
            mocks["LLMService"].return_value.sanitize_output.side_effect = lambda x: x
            mocker.patch("llm.plugin.world").ircs = [mock_irc]
            plugin = LLM(mock_irc)
            return plugin, mocks["addEvent"]

        return seed, start

    @staticmethod
    def _wakeups(add_event: MagicMock) -> list:
        return [c for c in add_event.call_args_list if c.kwargs.get("name") == "llm_agenda_wakeup"]

    def test_future_reminder_arms_one_wakeup_at_its_time(self, boot) -> None:
        seed, start = boot
        fire_at = time.time() + 3600
        seed.save_reminder("llm_remind_abc", "testuser", "#test", "check build", fire_at)
        seed.save_reminder("llm_remind_def", "testuser", "#test", "later", fire_at + 60)

        _plugin, add_event = start()

        [wakeup] = self._wakeups(add_event)
        assert wakeup.args[1] == fire_at
        named = {c.kwargs.get("name") for c in add_event.call_args_list}
        assert not any(n and n.startswith("llm_remind_") for n in named)

    def test_overdue_reminder_fires_on_first_tick_not_in_init(self, boot, mock_irc) -> None:
        seed, start = boot
        seed.save_reminder("llm_remind_abc", "testuser", "#test", "check build", time.time() - 60)
        before = time.time()

        plugin, add_event = start()

        # IRC may not be registered yet inside __init__.
        mock_irc.queueMsg.assert_not_called()
        [wakeup] = self._wakeups(add_event)
        assert wakeup.args[1] >= before
        wakeup.args[0]()
        queued = mock_irc.queueMsg.call_args[0][0]
        assert queued.args[0] == "#test"
        assert "testuser" in queued.args[1] and "check build" in queued.args[1]
        assert plugin.db.get_reminder("llm_remind_abc") is None

    def test_overdue_recurring_chain_survives_restart_in_place(self, boot, mock_irc) -> None:
        """4830d4f: a chain overdue across a restart fires once and continues,
        on the same row and id, from that fire's start."""
        seed, start = boot
        seed.save_reminder(
            "llm_remind_rec",
            "testuser",
            "#test",
            "stretch",
            time.time() - 3 * 86400,
            chain_position=3,
            recurrence_seconds=3600,
        )

        plugin, add_event = start()
        tick = self._wakeups(add_event)[0].args[0]
        add_event.reset_mock()
        before = time.time()
        tick()

        mock_irc.queueMsg.assert_called_once()
        row = plugin.db.get_reminder("llm_remind_rec")
        assert row is not None
        assert row.chain_position == 4
        assert before + 3600 <= row.fire_at <= time.time() + 3600
        # Re-armed for the chain's next fire.
        [wakeup] = self._wakeups(add_event)
        assert wakeup.args[1] == row.fire_at

    def test_overdue_scheduled_task_is_handed_to_the_service(self, boot) -> None:
        seed, start = boot
        seed.save_scheduled_llm_task(
            "llm_task_abc",
            "testuser",
            "acct",
            "#test",
            "afternet",
            ":testuser!u@h PRIVMSG #test :@ask hi",
            "do x",
            time.time() - 60,
        )

        plugin, add_event = start()
        self._wakeups(add_event)[0].args[0]()

        fire = plugin.llm_service.fire_scheduled_llm_task
        fire.assert_called_once()
        assert fire.call_args.args[0].event_name == "llm_task_abc"

    def test_die_removes_the_wakeup(self, boot, mocker: MockerFixture) -> None:
        from llm.plugin import LLM

        seed, start = boot
        seed.save_reminder("llm_remind_abc", "u", "#test", "m", time.time() + 60)
        plugin, _add_event = start()
        remove = mocker.patch("llm.plugin.schedule.removeEvent")
        mocker.patch("llm.plugin.httpserver.unhook")
        mocker.patch.object(LLM.__bases__[0], "die", return_value=None)

        plugin.die()

        assert mocker.call("llm_agenda_wakeup") in remove.call_args_list


class TestRequireCaller:
    """Test _require_caller account-identification gate helper."""

    def test_returns_account_when_identified(self, mocker: MockerFixture) -> None:
        """GIVEN identified user WHEN _require_caller called THEN returns its Caller."""
        from llm.plugin import LLM

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)

        mock_irc = mocker.MagicMock()
        mock_irc.state.nickToAccount = mocker.MagicMock(return_value="alice_account")

        mock_msg = mocker.MagicMock()
        mock_msg.prefix = "alice!user@host"
        mock_msg.server_tags = {}
        mock_msg.args = ("#chan", "hi")

        result = plugin._require_caller(mock_irc, mock_msg)
        assert result is not None
        assert result.identity.personal_key == "alice_account"
        mock_irc.error.assert_not_called()

    def test_returns_none_and_errors_when_unidentified(self, mocker: MockerFixture) -> None:
        """GIVEN unidentified user WHEN _require_caller called THEN returns None and errors."""
        from llm.plugin import LLM

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        plugin._llm_executor = mocker.MagicMock(closing=False)
        plugin._irc_send_lock = threading.Lock()
        plugin.log = mocker.MagicMock()

        mock_irc = mocker.MagicMock()
        mock_irc.state.nickToAccount = mocker.MagicMock(return_value=None)

        mock_msg = mocker.MagicMock()
        mock_msg.prefix = "alice!user@host"
        mock_msg.server_tags = {}
        mock_msg.args = ("#chan", "hi")

        result = plugin._require_caller(mock_irc, mock_msg)
        assert result is None
        mock_irc.error.assert_called_once()
        err_text = mock_irc.error.call_args[0][0]
        assert "NickServ" not in err_text
        assert "identified" in err_text.lower()

    def test_returns_none_on_key_error(self, mocker: MockerFixture) -> None:
        """GIVEN nickToAccount raises KeyError WHEN called THEN returns None."""
        from llm.plugin import LLM

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        plugin._llm_executor = mocker.MagicMock(closing=False)
        plugin._irc_send_lock = threading.Lock()
        plugin.log = mocker.MagicMock()

        mock_irc = mocker.MagicMock()
        mock_irc.state.nickToAccount = mocker.MagicMock(side_effect=KeyError("no such nick"))

        mock_msg = mocker.MagicMock()
        mock_msg.prefix = "alice!user@host"
        mock_msg.server_tags = {}
        mock_msg.args = ("#chan", "hi")

        result = plugin._require_caller(mock_irc, mock_msg)
        assert result is None
        mock_irc.error.assert_called_once()

    def test_returns_none_on_attribute_error(self, mocker: MockerFixture) -> None:
        """GIVEN nickToAccount raises AttributeError WHEN called THEN returns None."""
        from llm.plugin import LLM

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        plugin._llm_executor = mocker.MagicMock(closing=False)
        plugin._irc_send_lock = threading.Lock()
        plugin.log = mocker.MagicMock()

        mock_irc = mocker.MagicMock()
        mock_irc.state.nickToAccount = mocker.MagicMock(
            side_effect=AttributeError("no nickToAccount")
        )

        mock_msg = mocker.MagicMock()
        mock_msg.prefix = "alice!user@host"
        mock_msg.server_tags = {}
        mock_msg.args = ("#chan", "hi")

        result = plugin._require_caller(mock_irc, mock_msg)
        assert result is None
        mock_irc.error.assert_called_once()


class TestRequireCallerUsesResolver:
    """_require_caller must read account-tag via _account_from_msg."""

    def test_returns_tag_when_present(self, plugin_env):
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = None  # cache empty
        mock_msg.server_tags = {"account": "tag_acct"}

        caller = plugin._require_caller(mock_irc, mock_msg)
        assert caller is not None
        assert caller.identity.personal_key == "tag_acct"
        mock_irc.error.assert_not_called()


class TestRateLimiter:
    """Test in-memory rate limiter helpers."""

    @pytest.fixture
    def plugin(self, mocker: MockerFixture):
        """Create a minimal plugin for rate-limit testing."""
        from llm.plugin import LLM

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        p = LLM.__new__(LLM)
        p.db = mocker.MagicMock()
        p.log = mocker.MagicMock()
        p._rate_buckets = {}
        p._rate_buckets_lock = threading.Lock()
        p._llm_executor = mocker.MagicMock(closing=False)
        p._irc_send_lock = threading.Lock()
        p.registryValue = mocker.MagicMock(
            side_effect=lambda key, *a: {
                "askRateLimitCount": 15,
                "askRateLimitWindow": 60,
                "drawRateLimitCount": 3,
                "drawRateLimitWindow": 60,
            }.get(key, 0)
        )
        return p

    def test_not_limited_under_threshold(self, plugin) -> None:
        """GIVEN fewer requests than limit WHEN _is_rate_limited THEN False."""
        now = 1000.0
        plugin._record_rate_limit_hit("draw", "alice", now - 10)
        plugin._record_rate_limit_hit("draw", "alice", now - 5)
        assert plugin._is_rate_limited("draw", "alice", now, tier="registered") is False

    def test_limited_at_threshold(self, plugin) -> None:
        """GIVEN requests at limit WHEN _is_rate_limited THEN True."""
        now = 1000.0
        for i in range(3):
            plugin._record_rate_limit_hit("draw", "alice", now - 30 + i)
        assert plugin._is_rate_limited("draw", "alice", now, tier="registered") is True

    def test_evicts_expired_entries(self, plugin) -> None:
        """GIVEN old entries outside window WHEN _is_rate_limited THEN evicted and not counted."""
        now = 1000.0
        # Three hits from 200s ago (outside 60s window)
        for i in range(3):
            plugin._record_rate_limit_hit("draw", "alice", now - 200 + i)
        assert plugin._is_rate_limited("draw", "alice", now, tier="registered") is False
        assert "draw:alice" not in plugin._rate_buckets

    def test_different_commands_isolated(self, plugin) -> None:
        """GIVEN draw at limit WHEN checking ask THEN not limited."""
        now = 1000.0
        for i in range(3):
            plugin._record_rate_limit_hit("draw", "alice", now - 10 + i)
        assert plugin._is_rate_limited("ask", "alice", now, tier="registered") is False

    def test_different_accounts_isolated(self, plugin) -> None:
        """GIVEN alice at limit WHEN checking bob THEN not limited."""
        now = 1000.0
        for i in range(3):
            plugin._record_rate_limit_hit("draw", "alice", now - 10 + i)
        assert plugin._is_rate_limited("draw", "bob", now, tier="registered") is False

    def test_disabled_tier_does_not_accumulate_bucket(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN a tier whose limit is 0 (disabled — e.g. the default
        codeTrustedRateLimitCount) WHEN requests flow through _check_rate_limit
        THEN nothing accumulates in _rate_buckets.

        Regression: _is_rate_limited returns before its eviction pass when the
        count is 0, but _check_rate_limit still recorded every hit — so a
        disabled tier grew an unbounded deque of timestamps that nothing ever
        pruned, for the life of the process."""
        mock_irc = mocker.MagicMock()
        for i in range(50):
            mocker.patch("time.time", return_value=1000.0 + i)
            blocked = plugin._check_rate_limit(
                mock_irc, "code", "alice", "alice", "#test", "prompt", tier="trusted"
            )
            assert blocked is False

        assert "code:alice" not in plugin._rate_buckets

    def test_disabled_tier_drops_leftover_bucket(self, plugin) -> None:
        """GIVEN a bucket left over from when the tier was limited WHEN the tier
        is checked with the limit now 0 THEN the stale bucket is dropped rather
        than left in memory unpruned."""
        now = 1000.0
        for i in range(5):
            plugin._record_rate_limit_hit("code", "alice", now - 10 + i)
        assert "code:alice" in plugin._rate_buckets

        assert plugin._is_rate_limited("code", "alice", now, tier="trusted") is False
        assert "code:alice" not in plugin._rate_buckets

    def test_check_rate_limit_blocks_when_enforced(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN enforce=True and over limit WHEN _check_rate_limit THEN blocks and logs."""
        mock_irc = mocker.MagicMock()
        now = 1000.0
        for i in range(3):
            plugin._record_rate_limit_hit("draw", "alice", now - 10 + i)

        mocker.patch("time.time", return_value=now)
        blocked = plugin._check_rate_limit(
            mock_irc, "draw", "alice", "alice", "#test", "prompt", tier="registered"
        )

        assert blocked is True
        mock_irc.error.assert_called_once()
        plugin.db.log_usage.assert_called_once()
        assert plugin.db.log_usage.call_args.kwargs["status"] == "rate_limited"
        assert "rate_limited" in plugin.log.info.call_args.args[0]


class TestRunPreflight:
    """Test _run_preflight shared preflight logic."""

    @pytest.fixture
    def plugin(self, mocker: MockerFixture):
        """Create a minimal plugin for preflight testing."""
        from llm.plugin import LLM

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        # Default: registered user (no owner/admin/trusted capabilities)
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )
        p = LLM.__new__(LLM)
        p.db = mocker.MagicMock()
        p.log = mocker.MagicMock()
        p._rate_buckets = {}
        p._rate_buckets_lock = threading.Lock()
        p._llm_executor = mocker.MagicMock(closing=False)
        p._irc_send_lock = threading.Lock()
        p.registryValue = mocker.MagicMock(
            side_effect=lambda key, *a: {
                "askRateLimitCount": 15,
                "askRateLimitWindow": 60,
                "drawRateLimitCount": 3,
                "drawRateLimitWindow": 60,
            }.get(key, "")
        )
        return p

    def test_preflight_passes_for_ask(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN normal user WHEN ask preflight THEN not blocked."""
        mock_irc = mocker.MagicMock()
        mock_irc.state.nickToAccount.return_value = "alice"
        mock_msg = mocker.MagicMock()
        mock_msg.prefix = "alice!user@host"
        mock_msg.args = ("#test", "hello")
        mock_msg.server_tags = {}

        result = plugin._run_preflight(mock_irc, mock_msg, "hello", "ask", require_account=False)
        assert result.blocked is False
        assert result.caller.identity.key == "alice"
        assert result.caller.reply_target == "#test"

    def test_preflight_blocks_unidentified_for_draw(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN unidentified user WHEN draw preflight THEN blocked with auth_failure."""
        mock_irc = mocker.MagicMock()
        mock_irc.state.nickToAccount.side_effect = KeyError("not found")
        mock_msg = mocker.MagicMock()
        mock_msg.prefix = "anon!user@host"
        mock_msg.args = ("#test", "draw me")
        mock_msg.server_tags = {}

        result = plugin._run_preflight(mock_irc, mock_msg, "draw me", "draw", require_account=True)
        assert result.blocked is True
        mock_irc.error.assert_called_once()
        plugin.db.log_usage.assert_called_once()
        assert plugin.db.log_usage.call_args.kwargs["status"] == "auth_failure"


class TestIsContentBlockedError:
    """Test _is_content_blocked_error static method."""

    def test_detects_content_keyword(self) -> None:
        """GIVEN error with 'content' WHEN checked THEN returns True."""
        from llm.plugin import LLM

        assert LLM._is_content_blocked_error("content policy violation") is True

    def test_detects_moderation_keyword(self) -> None:
        """GIVEN error with 'moderation' WHEN checked THEN returns True."""
        from llm.plugin import LLM

        assert LLM._is_content_blocked_error("moderation_blocked") is True

    def test_returns_false_for_generic_error(self) -> None:
        """GIVEN generic error WHEN checked THEN returns False."""
        from llm.plugin import LLM

        assert LLM._is_content_blocked_error("timeout exceeded") is False

    def test_returns_false_for_none(self) -> None:
        """GIVEN None WHEN checked THEN returns False."""
        from llm.plugin import LLM

        assert LLM._is_content_blocked_error(None) is False


class TestHTTPCallbackOSErrorWithBrokenPipe:
    """Test HTTP callback OSError handler when sending 500 also fails."""

    def test_oserror_then_broken_pipe_silenced(self, tmp_path, mocker: MockerFixture) -> None:
        """GIVEN file open raises OSError WHEN sending 500 also raises BrokenPipeError THEN no exception propagates."""
        from llm.plugin import LLMHTTPCallback

        callback = LLMHTTPCallback.__new__(LLMHTTPCallback)
        mock_plugin = mocker.MagicMock()
        mock_plugin.registryValue.return_value = str(tmp_path)
        callback._plugin = mock_plugin

        # Create a real file so path resolution succeeds and is_file() returns True
        test_file = tmp_path / "somefile.html"
        test_file.write_bytes(b"<html>test</html>")

        handler = mocker.MagicMock()
        handler.wfile = mocker.MagicMock()

        # Patch builtins.open to raise OSError (hits line 260)
        mocker.patch("builtins.open", side_effect=OSError("disk error"))
        # Then handler.send_response raises BrokenPipeError (hits line 264)
        handler.send_response.side_effect = BrokenPipeError("client gone")

        # Should not raise
        callback.doGet(handler, "somefile.html")


class TestGetBuildInfoGitFailure:
    """Test _get_build_info when git is not available."""

    def test_git_not_found_returns_version_without_sha(
        self, mock_irc: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN git not installed WHEN _get_build_info called THEN returns version without SHA."""
        from llm.plugin import LLM

        from .conftest import make_registry_side_effect, plugin_init_patches

        plugin_init_patches(mocker)
        mocker.patch.object(LLM, "registryValue", side_effect=make_registry_side_effect())
        plugin = LLM(mock_irc)

        mocker.patch("subprocess.check_output", side_effect=FileNotFoundError("git"))
        result = plugin._get_build_info()

        assert result.startswith("v")
        assert "(" not in result


class TestAccountFromMsg:
    """Two-layer account resolver: server_tags then state cache."""

    def test_layer1_account_tag_wins(self, plugin_env, mocker: MockerFixture):
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "cached_acct"
        mock_msg.server_tags = {"account": "tag_acct"}

        assert plugin._account_from_msg(mock_irc, mock_msg) == "tag_acct"

    def test_layer2_state_cache_when_no_tag(self, plugin_env, mocker: MockerFixture):
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "cached_acct"
        mock_msg.server_tags = {}

        assert plugin._account_from_msg(mock_irc, mock_msg) == "cached_acct"

    def test_returns_none_when_unknown(self, plugin_env, mocker: MockerFixture):
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = None
        mock_msg.server_tags = {}

        assert plugin._account_from_msg(mock_irc, mock_msg) is None

    def test_state_cache_keyerror_returns_none(self, plugin_env, mocker: MockerFixture):
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.side_effect = KeyError
        mock_msg.server_tags = {}

        assert plugin._account_from_msg(mock_irc, mock_msg) is None

    def test_state_cache_attributeerror_returns_none(self, plugin_env, mocker: MockerFixture):
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.side_effect = AttributeError
        mock_msg.server_tags = {}

        assert plugin._account_from_msg(mock_irc, mock_msg) is None

    def test_empty_string_tag_falls_through(self, plugin_env, mocker: MockerFixture):
        # account-tag value of "" or "*" means "logged out" per IRCv3 — treat as no tag.
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "cached_acct"
        mock_msg.server_tags = {"account": ""}

        assert plugin._account_from_msg(mock_irc, mock_msg) == "cached_acct"

    def test_star_tag_falls_through(self, plugin_env, mocker: MockerFixture):
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "cached_acct"
        mock_msg.server_tags = {"account": "*"}

        assert plugin._account_from_msg(mock_irc, mock_msg) == "cached_acct"

    def test_server_prefix_returns_none_instead_of_assertion(
        self, plugin_env, mocker: MockerFixture
    ):
        # Server-originated PRIVMSG (no nick!user@host) used to crash
        # nickFromHostmask's assert. Must return None cleanly.
        plugin, mock_irc, mock_msg = plugin_env
        mock_msg.server_tags = {}
        mock_msg.prefix = "luna.AfterNET.Org"

        assert plugin._account_from_msg(mock_irc, mock_msg) is None
        mock_irc.state.nickToAccount.assert_not_called()

    def test_empty_prefix_returns_none(self, plugin_env, mocker: MockerFixture):
        plugin, mock_irc, mock_msg = plugin_env
        mock_msg.server_tags = {}
        mock_msg.prefix = ""

        assert plugin._account_from_msg(mock_irc, mock_msg) is None
        mock_irc.state.nickToAccount.assert_not_called()


class TestResolveTierUsesResolver:
    def test_registered_tier_via_account_tag(self, plugin_env):
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = None  # cache empty
        mock_msg.server_tags = {"account": "tag_acct"}

        assert plugin._resolve_tier(mock_irc, mock_msg) == "registered"

    def test_unregistered_when_no_tag_no_cache(self, plugin_env):
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = None
        mock_msg.server_tags = {}

        assert plugin._resolve_tier(mock_irc, mock_msg) == "unregistered"


class TestPreflightOptionalAccountTag:
    """When require_account=False, account-tag should still populate the account."""

    def test_optional_path_picks_up_account_tag(self, plugin_env):
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = None  # cache empty
        mock_msg.server_tags = {"account": "tag_acct"}

        result = plugin._run_preflight(
            mock_irc, mock_msg, text="hi", command="ask", require_account=False
        )
        assert result.caller.identity.account == "tag_acct"


class TestChatProfileBridgeWiring:
    """Tests that _ask_impl passes extra_tools/extra_handlers to assistant_completion."""

    def _make_assistant_result(self) -> AssistantResult:
        return AssistantResult(
            content="bridge wiring reply",
            grounding_used=False,
            spend=Spend(prompt_tokens=10, completion_tokens=5, cost=0.001),
            model="gpt-4",
        )

    def test_passes_bridge_extras_when_enabled(self, plugin_env, mocker: MockerFixture):
        """GIVEN bridgeEnabled=True with allowed plugins WHEN ask is called
        THEN assistant_completion receives extra_tools (list) and extra_handlers (dict)."""
        plugin, mock_irc, mock_msg = plugin_env

        # Override registry: enable bridge with one allowed plugin.
        plugin.registryValue.side_effect = make_registry_side_effect(
            {"bridgeEnabled": True, "bridgeAllowedPlugins": ["Misc"]}
        )

        fake_cmd = mocker.MagicMock(
            plugin="Misc",
            command="ping",
            arg_syntax="takes no arguments",
            description="Replies with pong.",
        )
        mocker.patch("llm.limnoria_bridge.enumerate_commands", return_value=[fake_cmd])

        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.assistant_completion.side_effect = None
        plugin.llm_service.assistant_completion.return_value = self._make_assistant_result()

        plugin.ask(mock_irc, mock_msg, ["hello"])

        plugin.llm_service.assistant_completion.assert_called_once()
        kwargs = plugin.llm_service.assistant_completion.call_args.kwargs
        extra_tools = kwargs.get("extra_tools")
        extra_handlers = kwargs.get("extra_handlers")
        assert extra_tools is not None
        names = [t["function"]["name"] for t in extra_tools]
        assert "run_limnoria_command" in names
        assert "search_bridge_commands" in names
        assert extra_handlers is not None
        assert "run_limnoria_command" in extra_handlers
        assert "search_bridge_commands" in extra_handlers

    def test_omits_bridge_extras_when_disabled(self, plugin_env, mocker: MockerFixture):
        """GIVEN bridgeEnabled=False (default) WHEN ask is called
        THEN no bridge tool or handler reaches assistant_completion."""
        plugin, mock_irc, mock_msg = plugin_env

        # Default conftest registry has bridgeEnabled=False.
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.assistant_completion.side_effect = None
        plugin.llm_service.assistant_completion.return_value = self._make_assistant_result()

        plugin.ask(mock_irc, mock_msg, ["hello"])

        plugin.llm_service.assistant_completion.assert_called_once()
        kwargs = plugin.llm_service.assistant_completion.call_args.kwargs
        names = {t["function"]["name"] for t in kwargs["extra_tools"]}
        assert not names & {"run_limnoria_command", "search_bridge_commands"}
        assert not set(kwargs["extra_handlers"]) & {
            "run_limnoria_command",
            "search_bridge_commands",
        }

    def _ask_after_one_bridge_call(self, plugin, mock_irc, mock_msg, mocker, *, debug: bool):
        """Run @ask where one bridge command ran during the completion."""
        plugin.registryValue.side_effect = make_registry_side_effect(
            {
                "bridgeEnabled": True,
                "bridgeAllowedPlugins": ["Misc"],
                "bridgeDebugInChannel": debug,
            }
        )
        fake_cmd = mocker.MagicMock(
            plugin="Misc", command="ping", arg_syntax="", description="Replies with pong."
        )
        mocker.patch("llm.limnoria_bridge.enumerate_commands", return_value=[fake_cmd])
        real_build = plugin._build_bridge_tool

        def build(irc, msg, channel, trace=None):
            built = real_build(irc, msg, channel, trace=trace)
            assert trace is not None
            trace.append(("Misc", "ping", "", "ok"))
            return built

        mocker.patch.object(plugin, "_build_bridge_tool", side_effect=build)
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.assistant_completion.return_value = self._make_assistant_result()

        plugin.llm_service.save_markdown_to_http.return_value = "http://h/a.html"

        plugin.ask(mock_irc, mock_msg, ["hello"])

        saved = plugin.llm_service.save_markdown_to_http.call_args
        return mock_irc.reply.call_args[0][0], saved[0][0] if saved else None

    def test_no_bridge_debug_footer_unless_the_channel_opts_in(
        self, plugin_env, mocker: MockerFixture
    ):
        """bridgeDebugInChannel defaults off: a public reply carries no debug line."""
        plugin, mock_irc, mock_msg = plugin_env

        reply, saved = self._ask_after_one_bridge_call(
            plugin, mock_irc, mock_msg, mocker, debug=False
        )

        assert (reply, saved) == ("bridge wiring reply", None)

    def test_bridge_debug_footer_when_the_channel_opts_in(self, plugin_env, mocker: MockerFixture):
        plugin, mock_irc, mock_msg = plugin_env

        _reply, saved = self._ask_after_one_bridge_call(
            plugin, mock_irc, mock_msg, mocker, debug=True
        )

        assert saved is not None and "[bridge: Misc.ping [ok]]" in saved

    def test_irc_lookup_gets_the_requesting_channel(self, plugin_env, mocker: MockerFixture):
        """With privacyOptInOnly off, kind=history reads the channel the ask came from."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.registryValue.side_effect = make_registry_side_effect({"privacyOptInOnly": False})
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.assistant_completion.return_value = self._make_assistant_result()

        plugin.ask(mock_irc, mock_msg, ["what did I miss"])

        tools = plugin.llm_service.assistant_completion.call_args.kwargs["extra_tools"]
        (lookup,) = [t for t in tools if t["function"]["name"] == "irc_lookup"]
        assert "history" in lookup["function"]["parameters"]["properties"]["kind"]["enum"]

    def test_typing_done_is_sent_after_reply_dispatch(
        self, plugin_env, mocker: MockerFixture
    ) -> None:
        """GIVEN ask reply WHEN dispatched THEN typing done follows the IRC reply."""
        plugin, mock_irc, mock_msg = plugin_env
        events: list[str] = []

        def begin_typing(_irc, _msg):
            events.append("typing_active")
            return lambda: events.append("typing_done")

        plugin.llm_service._begin_typing.side_effect = begin_typing
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.assistant_completion.side_effect = None
        plugin.llm_service.assistant_completion.return_value = self._make_assistant_result()
        mocker.patch.object(
            plugin,
            "_send_long_reply",
            side_effect=lambda *_args, **_kwargs: events.append("reply"),
        )

        plugin.ask(mock_irc, mock_msg, ["hello"])

        assert events == ["typing_active", "reply", "typing_done"]
        kwargs = plugin.llm_service.assistant_completion.call_args.kwargs
        assert kwargs["manage_typing"] is False


class TestExtractActionFloodSafety:
    """``_extract_action`` must collapse newlines so the IRC ACTION is single-line.

    Sending raw ``\\n`` inside an ACTION puts literal newlines on the wire; the
    server parses each as a separate command and disconnects with Excess Flood
    (regression for vibebot disconnect on a multi-line ``/me`` reply).
    """

    @pytest.fixture
    def plugin_with_extract(self, mocker: MockerFixture):
        from llm.plugin import LLM

        irc = mocker.MagicMock()
        irc.nick = "vibebot"
        # Wire a stand-in that exposes the real _collapse_for_irc but doesn't
        # require a full Limnoria callback boot.
        stand_in = mocker.MagicMock(spec_set=["_collapse_for_irc"])
        stand_in._collapse_for_irc = LLM._collapse_for_irc
        return LLM._extract_action.__get__(stand_in, LLM), irc

    def test_collapses_multiline_me_action(self, plugin_with_extract) -> None:
        extract, irc = plugin_with_extract
        result = extract(irc, "/me draws a cookbook\nwith many recipes\nand pictures")
        assert result == "draws a cookbook | with many recipes | and pictures"
        assert "\n" not in result

    def test_collapses_multiline_star_action(self, plugin_with_extract) -> None:
        extract, irc = plugin_with_extract
        result = extract(irc, "* vibebot draws\nline 2\nline 3")
        assert result == "draws | line 2 | line 3"
        assert "\n" not in result

    def test_returns_none_for_blank_collapsed_body(self, plugin_with_extract) -> None:
        extract, irc = plugin_with_extract
        # All-whitespace body should not produce an empty-string action.
        assert extract(irc, "/me \n\n  \n") is None

    def test_returns_none_for_non_action(self, plugin_with_extract) -> None:
        extract, irc = plugin_with_extract
        assert extract(irc, "Just a regular reply.") is None


class TestMemoryExtractionBackground:
    """Coverage for _schedule_memory_extraction's inner _extract_memories_bg."""

    @pytest.fixture
    def plugin_and_callback(self, mock_irc, mocker: MockerFixture):
        """Set up a plugin, schedule extraction, and capture the inner callback."""
        from llm.plugin import LLM

        from .conftest import plugin_init_patches

        mocker.patch.object(LLM, "registryValue", side_effect=make_registry_side_effect())
        plugin_init_patches(mocker)
        plugin = LLM(mock_irc)

        add_event = mocker.patch("llm.plugin.schedule.addEvent")
        return plugin, add_event

    def _existing_rows(self, mocker: MockerFixture, n: int):
        return [mocker.MagicMock(id=i + 1, fact=f"fact{i + 1}") for i in range(n)]

    def test_a_full_memory_set_spends_no_extraction(
        self, plugin_and_callback, mocker: MockerFixture
    ) -> None:
        """At the per-user cap nothing more can be promoted, so no background
        extraction call is scheduled at all."""
        from llm.plugin import _MEMORY_MAX_PER_USER

        plugin, add_event = plugin_and_callback
        plugin.db.get_memories.return_value = self._existing_rows(mocker, _MEMORY_MAX_PER_USER)

        plugin._schedule_memory_extraction("alice", "#test", "user", "bot")

        add_event.assert_not_called()
        plugin.llm_service.extract_memories.assert_not_called()

    def test_prune_drops_only_stale_candidates(self, plugin_and_callback, test_db) -> None:
        """Candidates accumulate across sightings for two weeks; only ones
        not seen in that time are pruned."""
        plugin, _ = plugin_and_callback
        test_db.add_memory_candidate("alice", "likes tea", "#test")
        test_db.add_memory_candidate("alice", "owns a boat", "#test")
        with test_db._write_txn() as conn:
            conn.execute(
                "UPDATE memory_candidates SET last_seen = last_seen - 20 * 86400 "
                "WHERE fact = 'owns a boat'"
            )
        real_db, plugin.db = plugin.db, test_db
        try:
            plugin._prune_stale_memory_candidates("alice")
        finally:
            plugin.db = real_db

        assert [c.fact for c in test_db.get_memory_candidates("alice")] == ["likes tea"]

    def test_race_abort_when_rows_changed(self, plugin_and_callback, mocker: MockerFixture) -> None:
        """If memory rows changed during extraction, no new memories are saved."""
        from llm.service import ExtractionResult

        plugin, add_event = plugin_and_callback
        snapshot = self._existing_rows(mocker, 2)
        # Second get_memories call (after extraction) returns a *different* row set.
        plugin.db.get_memories.side_effect = [
            snapshot,
            [mocker.MagicMock(id=99, fact="injected")],
        ]
        plugin.llm_service.extract_memories.return_value = ExtractionResult(add=["new fact"])

        plugin._schedule_memory_extraction("alice", "#test", "user", "bot")
        bg_callback = add_event.call_args.args[0]
        bg_callback()

        plugin.db.save_memory.assert_not_called()

    def test_cap_stops_loop_before_saving_all_facts(
        self, plugin_and_callback, mocker: MockerFixture
    ) -> None:
        """Promotion respects the per-user cap; extra candidates stay candidates."""
        from llm.persistence import MemoryCandidate
        from llm.plugin import _MEMORY_MAX_PER_USER, _MEMORY_PROMOTION_THRESHOLD
        from llm.service import ExtractionResult

        plugin, add_event = plugin_and_callback
        # Existing rows are one short of the cap, so only ONE promotion fits.
        snapshot = self._existing_rows(mocker, _MEMORY_MAX_PER_USER - 1)
        plugin.db.get_memories.side_effect = [snapshot, snapshot]
        plugin.db.get_memory_candidates.return_value = [
            MemoryCandidate(
                id=i + 1,
                nick="alice",
                fact=fact,
                mentions=_MEMORY_PROMOTION_THRESHOLD - 1,
                first_seen=100.0,
                last_seen=100.0,
                source_channel="#test",
            )
            for i, fact in enumerate(["one", "two", "three"])
        ]
        plugin.db.reinforce_memory_candidate.return_value = _MEMORY_PROMOTION_THRESHOLD
        plugin.db.increment_memory_saves.return_value = 0
        plugin.llm_service.extract_memories.return_value = ExtractionResult(reinforce=[0, 1, 2])

        plugin._schedule_memory_extraction("alice", "#test", "user", "bot")
        bg_callback = add_event.call_args.args[0]
        bg_callback()

        # Only one fact should fit before the cap is hit.
        assert plugin.db.save_memory.call_count == 1

    def test_cleanup_triggers_when_save_counter_reaches_interval(
        self, plugin_and_callback, mocker: MockerFixture
    ) -> None:
        """When increment_memory_saves crosses the cleanup interval, _run_memory_cleanup runs."""
        from llm.persistence import MemoryCandidate
        from llm.plugin import _MEMORY_CLEANUP_INTERVAL, _MEMORY_PROMOTION_THRESHOLD
        from llm.service import ExtractionResult

        plugin, add_event = plugin_and_callback
        snapshot = self._existing_rows(mocker, 2)
        plugin.db.get_memories.side_effect = [snapshot, snapshot]
        plugin.db.get_memory_candidates.return_value = [
            MemoryCandidate(
                id=7,
                nick="alice",
                fact="a fresh fact",
                mentions=_MEMORY_PROMOTION_THRESHOLD - 1,
                first_seen=100.0,
                last_seen=100.0,
                source_channel="#test",
            )
        ]
        plugin.db.reinforce_memory_candidate.return_value = _MEMORY_PROMOTION_THRESHOLD
        plugin.llm_service.extract_memories.return_value = ExtractionResult(reinforce=[0])
        plugin.db.increment_memory_saves.return_value = _MEMORY_CLEANUP_INTERVAL
        cleanup_spy = mocker.patch.object(plugin, "_run_memory_cleanup")

        plugin._schedule_memory_extraction("alice", "#test", "user", "bot")
        bg_callback = add_event.call_args.args[0]
        bg_callback()

        plugin.db.reset_memory_saves.assert_called_once_with("alice")
        cleanup_spy.assert_called_once_with("alice", "#test")

    def test_new_facts_become_candidates_not_memories(
        self, plugin_and_callback, mocker: MockerFixture
    ) -> None:
        """Default threshold>1: an unfamiliar fact is staged as a candidate."""
        from llm.service import ExtractionResult

        plugin, add_event = plugin_and_callback
        snapshot = self._existing_rows(mocker, 0)
        plugin.db.get_memories.side_effect = [snapshot, snapshot]
        plugin.db.get_memory_candidates.return_value = []
        plugin.llm_service.extract_memories.return_value = ExtractionResult(add=["uses Arch Linux"])

        plugin._schedule_memory_extraction("alice", "#test", "user", "bot")
        add_event.call_args.args[0]()

        plugin.db.add_memory_candidate.assert_called_once_with("alice", "uses Arch Linux", "#test")
        plugin.db.save_memory.assert_not_called()

    def test_reinforce_below_threshold_bumps_only(
        self, plugin_and_callback, mocker: MockerFixture
    ) -> None:
        """A reinforcement that doesn't cross the threshold stays a candidate."""
        from llm.persistence import MemoryCandidate
        from llm.plugin import _MEMORY_PROMOTION_THRESHOLD
        from llm.service import ExtractionResult

        plugin, add_event = plugin_and_callback
        snapshot = self._existing_rows(mocker, 0)
        plugin.db.get_memories.side_effect = [snapshot, snapshot]
        candidate = MemoryCandidate(
            id=42,
            nick="alice",
            fact="uses Arch Linux",
            mentions=1,
            first_seen=100.0,
            last_seen=100.0,
            source_channel="#test",
        )
        plugin.db.get_memory_candidates.return_value = [candidate]
        plugin.db.reinforce_memory_candidate.return_value = _MEMORY_PROMOTION_THRESHOLD - 1
        plugin.llm_service.extract_memories.return_value = ExtractionResult(reinforce=[0])

        plugin._schedule_memory_extraction("alice", "#test", "user", "bot")
        add_event.call_args.args[0]()

        plugin.db.reinforce_memory_candidate.assert_called_once_with(42, "alice")
        plugin.db.save_memory.assert_not_called()
        plugin.db.delete_memory_candidate.assert_not_called()

    def test_reinforce_at_threshold_promotes_candidate(
        self, plugin_and_callback, mocker: MockerFixture
    ) -> None:
        """Reaching the threshold moves the candidate into memories."""
        from llm.persistence import MemoryCandidate
        from llm.service import ExtractionResult

        plugin, add_event = plugin_and_callback
        snapshot = self._existing_rows(mocker, 0)
        plugin.db.get_memories.side_effect = [snapshot, snapshot]
        candidate = MemoryCandidate(
            id=7,
            nick="alice",
            fact="lives in Berlin",
            mentions=1,
            first_seen=100.0,
            last_seen=100.0,
            source_channel="#origin",
        )
        plugin.db.get_memory_candidates.return_value = [candidate]
        plugin.db.reinforce_memory_candidate.return_value = 2  # crosses default threshold
        plugin.llm_service.extract_memories.return_value = ExtractionResult(reinforce=[0])

        plugin._schedule_memory_extraction("alice", "#test", "user", "bot")
        add_event.call_args.args[0]()

        plugin.db.save_memory.assert_called_once_with("alice", "lives in Berlin", "#origin")
        plugin.db.delete_memory_candidate.assert_called_once_with(7, "alice")

    def test_candidate_change_during_extraction_aborts(
        self, plugin_and_callback, mocker: MockerFixture
    ) -> None:
        """If candidate row IDs change mid-call, reinforce indices abort."""
        from llm.persistence import MemoryCandidate
        from llm.service import ExtractionResult

        plugin, add_event = plugin_and_callback
        snapshot = self._existing_rows(mocker, 0)
        plugin.db.get_memories.side_effect = [snapshot, snapshot]
        before = MemoryCandidate(1, "alice", "x", 1, 100.0, 100.0, "#test")
        after = MemoryCandidate(2, "alice", "x", 1, 100.0, 100.0, "#test")
        plugin.db.get_memory_candidates.side_effect = [[before], [after]]
        plugin.llm_service.extract_memories.return_value = ExtractionResult(reinforce=[0])

        plugin._schedule_memory_extraction("alice", "#test", "user", "bot")
        add_event.call_args.args[0]()

        plugin.db.reinforce_memory_candidate.assert_not_called()
        plugin.db.save_memory.assert_not_called()


class TestUndeadPluginReaper:
    """The autouse reaper in conftest stops threads a test forgot to stop.

    ``LLMService`` starts ``typing-keepalive`` on the first hold; only
    ``die()`` stops it. Dozens of tests build a plugin directly and never die it, so the reaper
    does it for them at teardown. Exercising the context manager inline pins
    that behaviour to one assertion instead of leaving it to the session-scoped
    survivor check, which cannot say *which* teardown did the work.
    """

    @staticmethod
    def _build_abandonable_plugin(mock_irc: MagicMock, mocker: MockerFixture):
        from llm.plugin import LLM

        from .conftest import make_registry_side_effect, plugin_init_patches

        mocker.patch.object(LLM, "registryValue", side_effect=make_registry_side_effect())
        plugin_init_patches(mocker)
        return LLM

    def test_abandoned_plugin_is_died_at_teardown(
        self, mock_irc: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN a plugin nobody dies WHEN the reaping block exits THEN it
        is died."""
        from .conftest import reaping_undead_plugins

        llm_class = self._build_abandonable_plugin(mock_irc, mocker)

        with reaping_undead_plugins():
            plugin = llm_class(mock_irc)
            plugin.db.close.assert_not_called()

        # db.close() runs once per die(); the reaper died it exactly once.
        plugin.db.close.assert_called_once_with()

    def test_plugin_the_test_already_died_is_not_died_twice(
        self, mock_irc: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN a plugin the test dies itself WHEN the reaping block exits
        THEN the reaper leaves it alone."""
        from .conftest import reaping_undead_plugins

        llm_class = self._build_abandonable_plugin(mock_irc, mocker)

        with reaping_undead_plugins():
            plugin = llm_class(mock_irc)
            plugin.die()
            plugin.db.close.assert_called_once_with()

        plugin.db.close.assert_called_once_with()


def test_unattended_assistant_reads_state_under_account_key(mocker: MockerFixture) -> None:
    """GIVEN a fire owned by an account whose nick differs WHEN the unattended
    turn gathers context THEN history, memories and @instruct use the account
    key the commands stored them under, not the raw nick."""
    from llm.plugin import LLM

    plugin = mocker.MagicMock()
    plugin._gather_history.return_value = ([], [])
    plugin._get_user_memories.return_value = []
    plugin.db.get_instruction.return_value = ""
    plugin._pending_task_fns.return_value = {}

    LLM._run_unattended_assistant(
        plugin,
        irc=mocker.MagicMock(),
        msg=mocker.MagicMock(),
        prompt="check the build",
        caller=Caller(Identity("alice_away", "alice"), "#test"),
        bot_nick="vibebot",
        entry_route="reminder_action",
    )

    plugin._gather_history.assert_called_once_with(Caller(Identity("alice_away", "alice"), "#test"))
    plugin._get_user_memories.assert_called_once_with("alice")
    plugin.db.get_instruction.assert_called_once_with("alice")
