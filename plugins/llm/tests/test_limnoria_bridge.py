"""Tests for the Limnoria tool bridge."""

from __future__ import annotations

import pytest


def test_module_exposes_deny_lists_and_dataclass():
    from llm import limnoria_bridge as lb

    assert isinstance(lb.DENY_PLUGINS, frozenset)
    assert "LLM" in lb.DENY_PLUGINS
    assert "Owner" in lb.DENY_PLUGINS
    assert "Admin" in lb.DENY_PLUGINS
    assert "Config" in lb.DENY_PLUGINS
    assert "Channel" in lb.DENY_PLUGINS
    assert "User" in lb.DENY_PLUGINS

    assert isinstance(lb.DENY_COMMANDS, frozenset)
    assert ("misc", "more") in lb.DENY_COMMANDS
    assert ("misc", "clearmores") in lb.DENY_COMMANDS
    assert ("web", "fetch") in lb.DENY_COMMANDS
    assert ("utilities", "apply") in lb.DENY_COMMANDS
    assert ("utilities", "let") in lb.DENY_COMMANDS

    cmd = lb.BridgeCommand(
        plugin="Misc", command="ping", arg_syntax="", description="takes no arguments"
    )
    assert cmd.plugin == "Misc"
    assert cmd.command == "ping"


def test_mutating_commands_covers_default_allowlist_writes():
    """Every mutating command in the Phase 2 Task 2 default allowlist must be
    in MUTATING_COMMANDS. Reads must NOT be in it. Tuples are
    (canonical_plugin_lowercase, leaf_lowercase) — same shape as DENY_COMMANDS."""
    from llm import limnoria_bridge as lb

    assert isinstance(lb.MUTATING_COMMANDS, frozenset)

    expected_mutating = {
        ("misc", "tell"),
        ("misc", "noticetell"),
        ("later", "tell"),
        ("later", "remove"),
        ("later", "undo"),
        ("note", "send"),
        ("note", "reply"),
        ("note", "unsend"),
        ("karma", "clear"),
        ("karma", "dump"),
        ("karma", "load"),
        ("quotegrabs", "grab"),
        ("quotegrabs", "ungrab"),
        ("rss", "add"),
        ("rss", "remove"),
        ("rss", "rss"),  # update_feed_if_needed → announce_feed → IRC writes
        ("rss", "info"),  # same update_feed_if_needed side effect
    }
    assert expected_mutating <= lb.MUTATING_COMMANDS

    # Reads in the same plugins must NOT be classified mutating.
    expected_read_only = {
        ("misc", "ping"),
        ("misc", "last"),
        ("misc", "version"),
        ("time", "time"),
        ("math", "calc"),
        ("utilities", "echo"),
        ("seen", "seen"),
        ("seen", "last"),
        ("web", "title"),
        ("later", "notes"),
        ("note", "search"),
        ("note", "list"),
        ("note", "note"),
        ("note", "next"),
        ("karma", "karma"),
        ("karma", "most"),
        ("quotegrabs", "quote"),
        ("quotegrabs", "random"),
        ("ddg", "search"),
    }
    assert expected_read_only.isdisjoint(lb.MUTATING_COMMANDS)


def test_mutating_commands_covers_forward_look_writes():
    """Quote/Todo/Factoids/Scheduler are not yet in the default allowlist but
    we classify them now so the gate is correct when they're added later."""
    from llm import limnoria_bridge as lb

    expected_mutating = {
        ("quote", "add"),
        ("quote", "remove"),
        ("quote", "change"),
        ("quote", "replace"),
        ("todo", "add"),
        ("todo", "remove"),
        ("todo", "setpriority"),
        ("todo", "change"),
        ("factoids", "learn"),
        ("factoids", "alias"),
        ("factoids", "lock"),
        ("factoids", "unlock"),
        ("factoids", "forget"),
        ("factoids", "change"),
        ("factoids", "whatis"),  # _updateRank writes when keepRankInfo=True (default)
        ("scheduler", "add"),
        ("scheduler", "remind"),
        ("scheduler", "remove"),
        ("scheduler", "repeat"),
    }
    assert expected_mutating <= lb.MUTATING_COMMANDS

    expected_read_only = {
        ("quote", "search"),
        ("quote", "get"),
        ("quote", "stats"),
        ("quote", "random"),
        ("todo", "todo"),
        ("todo", "search"),
        ("factoids", "random"),
        ("factoids", "info"),
        ("factoids", "rank"),
        ("factoids", "search"),
        ("scheduler", "list"),
    }
    assert expected_read_only.isdisjoint(lb.MUTATING_COMMANDS)


def test_network_connection_management_is_denied():
    """Network is allowlisted for whois/whowas/latency/uptime; the leaves
    that change the bot's connections or re-dispatch arbitrary commands
    (``command``/``cmdall`` are the same shape as Utilities.apply) are
    denied at the bridge layer regardless of the caller's capabilities."""
    from llm import limnoria_bridge as lb

    for leaf in ("connect", "disconnect", "reconnect", "command", "cmdall", "authenticate"):
        assert ("network", leaf) in lb.DENY_COMMANDS, leaf
    for leaf in ("whois", "whowas", "latency", "uptime", "networks", "driver", "capabilities"):
        assert ("network", leaf) not in lb.DENY_COMMANDS, leaf
        assert ("network", leaf) not in lb.MUTATING_COMMANDS, leaf


def test_buffering_proxy_does_not_queue_irc_traffic(mocker):
    from llm.limnoria_bridge import BufferingIrcProxy

    real_irc = mocker.MagicMock()
    real_irc.network = "testnet"
    msg = mocker.MagicMock()
    msg.args = ("#test", "trigger")
    msg.channel = "#test"

    proxy = BufferingIrcProxy(real_irc, msg)
    proxy.reply("captured")

    real_irc.queueMsg.assert_not_called()
    real_irc.sendMsg.assert_not_called()


def test_buffering_proxy_error_raise_true_still_raises(mocker):
    """error(Raise=True) must raise callbacks.Error so command flow stops.

    Many plugins use error(..., Raise=True) for early-exit; swallowing it
    silently lets the command continue past what should be a hard stop.
    """
    from llm.limnoria_bridge import BufferingIrcProxy
    from supybot import callbacks

    real_irc = mocker.MagicMock()
    real_irc.network = "testnet"
    msg = mocker.MagicMock()
    msg.args = ("#test", "trigger")
    msg.channel = "#test"

    proxy = BufferingIrcProxy(real_irc, msg)
    with pytest.raises(callbacks.Error):
        proxy.error("nope", Raise=True)
    # Buffer still captured the text before raising.
    assert proxy.buffer == ["nope"]


def _stub_callback(mocker, name, canonical=None, commands=None, docstrings=None):
    """Build a fake Limnoria plugin callback with controllable commands."""
    cb = mocker.MagicMock()
    cb.name.return_value = name
    cb.canonicalName.return_value = canonical or name.lower()
    cb.listCommands.return_value = list(commands or [])

    docs = docstrings or {}

    def _get_method(path):
        leaf = path[-1] if isinstance(path, list) else path
        method = mocker.MagicMock()
        method.__doc__ = docs.get(leaf, "")
        return method

    cb.getCommandMethod.side_effect = _get_method
    cb.isCommandMethod.side_effect = lambda c: c in (commands or [])
    return cb


def _fake_irc_with_callbacks(mocker, callbacks_list, network="testnet"):
    irc = mocker.MagicMock()
    irc.callbacks = list(callbacks_list)
    irc.network = network
    return irc


def _fake_msg(mocker, channel="#test", prefix="testnick!user@host"):
    msg = mocker.MagicMock()
    msg.prefix = prefix
    msg.channel = channel
    msg.args = (channel, "trigger")
    return msg


def test_enumerate_yields_command_when_authorized(mocker):
    from llm import limnoria_bridge as lb

    cb = _stub_callback(
        mocker,
        "Misc",
        commands=["ping"],
        docstrings={"ping": "takes no arguments\n\nReplies with pong."},
    )
    irc = _fake_irc_with_callbacks(mocker, [cb])
    msg = _fake_msg(mocker)

    mocker.patch.object(lb.callbacks, "checkCommandCapability", return_value=False)

    result = list(lb.enumerate_commands(irc, msg, frozenset({"Misc"})))

    assert len(result) == 1
    assert result[0].plugin == "Misc"
    assert result[0].command == "ping"
    assert result[0].arg_syntax == "takes no arguments"
    assert "Replies with pong." in result[0].description


def test_enumerate_skips_nested_command_leaf_without_crashing(mocker):
    """GIVEN a nested command group (e.g. RSS 'announce add') WHEN enumerating THEN skip it, don't raise.

    cb.listCommands() surfaces nested commands as multi-word leaves whose list
    form is not the plugin's canonical name, so cb.getCommandMethod([leaf])
    raises an AssertionError in Limnoria. Unguarded, that raise aborts the
    entire enumeration — and with it the whole @ask request, since a plugin
    with nested groups (RSS) is in the default allowlist. The bad leaf must be
    skipped while sibling top-level commands still enumerate.
    """
    from llm import limnoria_bridge as lb

    cb = mocker.MagicMock()
    cb.name.return_value = "StubPlugin"
    cb.canonicalName.return_value = "stubplugin"
    cb.listCommands.return_value = ["info", "group sub"]

    def _get_method(path):
        leaf = path[-1] if isinstance(path, list) else path
        if leaf == "group sub":
            raise AssertionError("'group sub' is not the canonical command name")
        method = mocker.MagicMock()
        method.__doc__ = "takes no arguments\n\nShows info."
        return method

    cb.getCommandMethod.side_effect = _get_method
    irc = _fake_irc_with_callbacks(mocker, [cb])
    msg = _fake_msg(mocker)
    mocker.patch.object(lb.callbacks, "checkCommandCapability", return_value=False)

    result = list(lb.enumerate_commands(irc, msg, frozenset({"StubPlugin"})))

    assert [c.command for c in result] == ["info"]


def test_enumerate_skips_deny_plugin_even_if_allowed(mocker):
    """Owner is hard-deny; explicitly adding it to the allowlist must not expose it."""
    from llm import limnoria_bridge as lb

    cb = _stub_callback(mocker, "Owner", commands=["load"])
    irc = _fake_irc_with_callbacks(mocker, [cb])
    msg = _fake_msg(mocker)
    mocker.patch.object(lb.callbacks, "checkCommandCapability", return_value=False)

    result = list(lb.enumerate_commands(irc, msg, frozenset({"Owner"})))

    assert result == []


def test_enumerate_skips_plugin_not_in_allowlist(mocker):
    from llm import limnoria_bridge as lb

    cb = _stub_callback(mocker, "Misc", commands=["ping"], docstrings={"ping": "x"})
    irc = _fake_irc_with_callbacks(mocker, [cb])
    msg = _fake_msg(mocker)
    mocker.patch.object(lb.callbacks, "checkCommandCapability", return_value=False)

    result = list(lb.enumerate_commands(irc, msg, frozenset()))  # empty allowlist

    assert result == []


def test_enumerate_skips_deny_command(mocker):
    """Web is allowed by operator, but Web.fetch is in DENY_COMMANDS."""
    from llm import limnoria_bridge as lb

    cb = _stub_callback(
        mocker,
        "Web",
        canonical="web",
        commands=["fetch", "urlquote"],
        docstrings={"fetch": "<url>", "urlquote": "<text>"},
    )
    irc = _fake_irc_with_callbacks(mocker, [cb])
    msg = _fake_msg(mocker)
    mocker.patch.object(lb.callbacks, "checkCommandCapability", return_value=False)

    result = list(lb.enumerate_commands(irc, msg, frozenset({"Web"})))

    leaves = {c.command for c in result}
    assert leaves == {"urlquote"}  # fetch is denied; urlquote (no network) passes


def test_enumerate_skips_lacking_capability(mocker):
    """Stub plugin (NOT in DENY_PLUGINS) whose command is anti-capability blocked."""
    from llm import limnoria_bridge as lb

    cb = _stub_callback(
        mocker,
        "StubPlugin",
        canonical="stubplugin",
        commands=["restricted", "open"],
        docstrings={"restricted": "x", "open": "y"},
    )
    irc = _fake_irc_with_callbacks(mocker, [cb])
    msg = _fake_msg(mocker)

    # Capability check returns truthy ("anti-cap-name") for `restricted`,
    # False (allowed) for `open`.
    def _check(_msg, _cb, name):
        return "stubplugin.restricted" if name == "restricted" else False

    mocker.patch.object(lb.callbacks, "checkCommandCapability", side_effect=_check)

    result = list(lb.enumerate_commands(irc, msg, frozenset({"StubPlugin"})))

    leaves = {c.command for c in result}
    assert leaves == {"open"}


def test_enumerate_passes_string_form_to_capability_check(mocker):
    """Regression: list form [cmd] triggers AssertionError in Limnoria.

    See callbacks.py:443-445 — checkCommandCapability asserts that
    list-form names start with the plugin's canonical name. We pass the
    string form to mirror _callCommand's leaf-check pattern at line 1591.
    """
    from llm import limnoria_bridge as lb

    cb = _stub_callback(mocker, "Misc", commands=["ping"], docstrings={"ping": "x"})
    irc = _fake_irc_with_callbacks(mocker, [cb])
    msg = _fake_msg(mocker)

    seen = []

    def _check(_msg, _cb, name):
        seen.append(name)
        return False

    mocker.patch.object(lb.callbacks, "checkCommandCapability", side_effect=_check)

    list(lb.enumerate_commands(irc, msg, frozenset({"Misc"})))

    assert seen == ["ping"]
    assert all(isinstance(n, str) for n in seen)


def test_enumerate_skips_mutating_commands(mocker):
    """MUTATING_COMMANDS leaves are filtered out even if their plugin is
    allowlisted."""
    from llm import limnoria_bridge as lb

    cb = _stub_callback(
        mocker,
        "Later",
        canonical="later",
        commands=["tell", "notes", "remove", "undo"],
        docstrings={
            "tell": "<nick> <text>",
            "notes": "takes no arguments",
            "remove": "<id>",
            "undo": "takes no arguments",
        },
    )
    irc = _fake_irc_with_callbacks(mocker, [cb])
    msg = _fake_msg(mocker)
    mocker.patch.object(lb.callbacks, "checkCommandCapability", return_value=False)

    result = list(lb.enumerate_commands(irc, msg, frozenset({"Later"})))

    leaves = {c.command for c in result}
    assert leaves == {"notes"}  # tell, remove, undo all in MUTATING_COMMANDS


def test_dispatch_unknown_plugin(mocker):
    from llm import limnoria_bridge as lb

    irc = mocker.MagicMock()
    irc.getCallback.return_value = None
    msg = _fake_msg(mocker)

    out = lb.dispatch(irc, msg, plugin="Nope", command="x", arg_string="")
    assert out == {"error": "unknown plugin: Nope"}


def test_dispatch_deny_plugin_blocks_call(mocker):
    from llm import limnoria_bridge as lb

    cb = _stub_callback(mocker, "Owner", commands=["load"])
    irc = mocker.MagicMock()
    irc.getCallback.return_value = cb
    msg = _fake_msg(mocker)

    out = lb.dispatch(irc, msg, plugin="Owner", command="load", arg_string="Foo")
    assert out == {"error": "denied: Owner.load"}


def test_dispatch_deny_command_blocks_call(mocker):
    from llm import limnoria_bridge as lb

    cb = _stub_callback(mocker, "Web", canonical="web", commands=["fetch"])
    irc = mocker.MagicMock()
    irc.getCallback.return_value = cb
    msg = _fake_msg(mocker)

    out = lb.dispatch(irc, msg, plugin="Web", command="fetch", arg_string="http://x")
    assert out == {"error": "denied: Web.fetch"}


def test_dispatch_blocks_web_ssrf_siblings_without_calling_command(mocker):
    """Defense in depth: dispatch rejects the SSRF siblings and never reaches
    Limnoria's command path."""
    from llm import limnoria_bridge as lb

    for leaf in ("location", "headers", "doctype", "size", "title"):
        cb = _stub_callback(mocker, "Web", canonical="web", commands=[leaf])
        irc = mocker.MagicMock()
        irc.getCallback.return_value = cb
        msg = _fake_msg(mocker)

        out = lb.dispatch(
            irc,
            msg,
            plugin="Web",
            command=leaf,
            arg_string="http://169.254.169.254/latest/meta-data/",
        )
        assert out == {"error": f"denied: Web.{leaf}"}
        cb._callCommand.assert_not_called()


def test_dispatch_unknown_command(mocker):
    from llm import limnoria_bridge as lb

    cb = _stub_callback(mocker, "Misc", commands=["ping"])
    irc = mocker.MagicMock()
    irc.getCallback.return_value = cb
    msg = _fake_msg(mocker)

    out = lb.dispatch(irc, msg, plugin="Misc", command="bogus", arg_string="")
    assert out == {"error": "unknown command: Misc.bogus"}


def test_dispatch_capability_denied(mocker):
    from llm import limnoria_bridge as lb

    cb = _stub_callback(mocker, "Misc", commands=["ping"])
    irc = mocker.MagicMock()
    irc.getCallback.return_value = cb
    msg = _fake_msg(mocker)
    mocker.patch.object(lb.callbacks, "checkCommandCapability", return_value="anti.cap")

    out = lb.dispatch(irc, msg, plugin="Misc", command="ping", arg_string="")
    assert out == {"error": "not permitted: Misc.ping"}


def test_dispatch_captures_reply(mocker):
    from llm import limnoria_bridge as lb

    cb = _stub_callback(mocker, "Misc", commands=["ping"])

    def _fake_call(command, proxy, _msg, _tokens):
        proxy.reply("pong")

    cb._callCommand.side_effect = _fake_call

    irc = mocker.MagicMock()
    irc.getCallback.return_value = cb
    irc.network = "testnet"
    msg = _fake_msg(mocker)
    mocker.patch.object(lb.callbacks, "checkCommandCapability", return_value=False)
    mocker.patch.object(lb.callbacks, "tokenize", return_value=[])

    out = lb.dispatch(irc, msg, plugin="Misc", command="ping", arg_string="")
    assert out == {"status": "ok", "reply": "pong"}


def test_dispatch_refuses_loaded_plugin_outside_allowlist(mocker):
    """GIVEN bridgeAllowedPlugins Time WHEN the model calls a loaded Alias.add
    THEN dispatch refuses it — the allowlist binds dispatch, not just the menu."""
    from llm import limnoria_bridge as lb

    cb = _stub_callback(mocker, "Alias", commands=["add"])
    irc = mocker.MagicMock()
    irc.getCallback.return_value = cb
    msg = _fake_msg(mocker)
    mocker.patch.object(lb.callbacks, "checkCommandCapability", return_value=False)

    out = lb.dispatch(
        irc,
        msg,
        plugin="Alias",
        command="add",
        arg_string="x echo y",
        allowed=frozenset({"Time"}),
    )

    assert out == {"error": "denied: Alias.add"}
    cb._callCommand.assert_not_called()


def test_dispatch_empty_allowlist_means_curated_default(mocker):
    """GIVEN an empty allowlist WHEN dispatching THEN the curated default
    applies, as in enumerate: Misc runs, Anonymous is refused."""
    from llm import limnoria_bridge as lb

    mocker.patch.object(lb.callbacks, "checkCommandCapability", return_value=False)
    mocker.patch.object(lb.callbacks, "tokenize", return_value=[])
    msg = _fake_msg(mocker)

    anon = _stub_callback(mocker, "Anonymous", commands=["say"])
    irc = mocker.MagicMock()
    irc.getCallback.return_value = anon
    out = lb.dispatch(irc, msg, plugin="Anonymous", command="say", arg_string="#c hi")
    assert out == {"error": "denied: Anonymous.say"}

    misc = _stub_callback(mocker, "Misc", commands=["ping"])
    misc._callCommand.side_effect = lambda _c, proxy, _m, _t: proxy.reply("pong")
    irc.getCallback.return_value = misc
    out = lb.dispatch(irc, msg, plugin="misc", command="ping", arg_string="", allowed=frozenset())
    assert out == {"status": "ok", "reply": "pong"}


def test_dispatch_does_not_log_raw_arg_string_at_info(mocker):
    """LLM-generated arg_string can carry secrets (e.g. a URL with an API key);
    it must never be logged at INFO. Only its length is INFO-logged."""
    from llm import limnoria_bridge as lb

    cb = _stub_callback(mocker, "Misc", commands=["ping"])
    cb._callCommand.side_effect = lambda command, proxy, _m, _t: proxy.reply("pong")
    irc = mocker.MagicMock()
    irc.getCallback.return_value = cb
    irc.network = "testnet"
    msg = _fake_msg(mocker)
    mocker.patch.object(lb.callbacks, "checkCommandCapability", return_value=False)
    mocker.patch.object(lb.callbacks, "tokenize", return_value=[])
    log = mocker.patch.object(lb, "_log")

    secret = "https://api.example.com/v1?key=sk-SUPER-SECRET-12345"  # noqa: S105
    lb.dispatch(irc, msg, plugin="Misc", command="ping", arg_string=secret)

    info_calls = [str(c.args) for c in log.info.call_args_list]
    assert not any(secret in c for c in info_calls), (
        f"raw arg_string leaked into INFO logs: {info_calls}"
    )


def test_dispatch_passes_command_as_list_and_tokens_positionally(mocker):
    """Regression: _callCommand requires a list-form command and positional tokens.

    Keyword `args=tokens` ends up in **kwargs (the wrap() spec receives an
    empty positional args list), breaking argument parsing.
    """
    from llm import limnoria_bridge as lb

    cb = _stub_callback(mocker, "Misc", commands=["ping"])
    cb._callCommand.return_value = None
    irc = mocker.MagicMock()
    irc.getCallback.return_value = cb
    irc.network = "testnet"
    msg = _fake_msg(mocker)
    mocker.patch.object(lb.callbacks, "checkCommandCapability", return_value=False)
    mocker.patch.object(lb.callbacks, "tokenize", return_value=["arg1", "arg2"])

    lb.dispatch(irc, msg, plugin="Misc", command="ping", arg_string="arg1 arg2")

    args, kwargs = cb._callCommand.call_args
    assert args[0] == ["ping"]
    # args = (command_list, irc, msg, tokens)
    assert args[3] == ["arg1", "arg2"]
    assert "args" not in kwargs


def test_dispatch_plugin_exception_returns_generic_error(mocker):
    """A plugin-level exception must NOT leak its detail to the model (it may
    carry hostnames/addresses/stack hints) — return a generic error."""
    from llm import limnoria_bridge as lb

    cb = _stub_callback(mocker, "Misc", commands=["ping"])
    cb._callCommand.side_effect = RuntimeError("could not connect to db at pg.internal:5432")
    irc = mocker.MagicMock()
    irc.getCallback.return_value = cb
    irc.network = "testnet"
    msg = _fake_msg(mocker)
    mocker.patch.object(lb.callbacks, "checkCommandCapability", return_value=False)
    mocker.patch.object(lb.callbacks, "tokenize", return_value=[])

    out = lb.dispatch(irc, msg, plugin="Misc", command="ping", arg_string="")
    assert out == {"error": "command failed"}
    assert "pg.internal" not in repr(out)


def test_dispatch_exception_preserves_buffered_partial_output(mocker):
    """Output the plugin buffered before raising must not be lost — it is
    returned as partial_output alongside the generic error."""
    from llm import limnoria_bridge as lb

    cb = _stub_callback(mocker, "Misc", commands=["ping"])

    def _partial_then_boom(_command, proxy, _msg, _tokens):
        proxy.reply("first line of output")
        raise RuntimeError("then it broke")

    cb._callCommand.side_effect = _partial_then_boom
    irc = mocker.MagicMock()
    irc.getCallback.return_value = cb
    irc.network = "testnet"
    msg = _fake_msg(mocker)
    mocker.patch.object(lb.callbacks, "checkCommandCapability", return_value=False)
    mocker.patch.object(lb.callbacks, "tokenize", return_value=[])

    out = lb.dispatch(irc, msg, plugin="Misc", command="ping", arg_string="")
    assert out["error"] == "command failed"
    assert out["partial_output"] == "first line of output"


def test_dispatch_malformed_args_returns_error_envelope(mocker):
    """tokenize() raises SyntaxError on malformed brackets/pipes — the
    bridge must catch it and return an error envelope, not propagate."""
    from llm import limnoria_bridge as lb

    cb = _stub_callback(mocker, "Misc", commands=["ping"])
    irc = mocker.MagicMock()
    irc.getCallback.return_value = cb
    irc.network = "testnet"
    msg = _fake_msg(mocker)
    mocker.patch.object(lb.callbacks, "checkCommandCapability", return_value=False)
    mocker.patch.object(lb.callbacks, "tokenize", side_effect=SyntaxError("unmatched bracket"))

    out = lb.dispatch(irc, msg, plugin="Misc", command="ping", arg_string="[oops")
    assert out == {"error": "unmatched bracket"}
    cb._callCommand.assert_not_called()


def test_dispatch_rejects_mutating(mocker):
    """Dispatching a MUTATING_COMMANDS leaf returns {"error": "denied: write commands disabled"}."""
    from llm import limnoria_bridge as lb

    cb = _stub_callback(mocker, "Later", canonical="later", commands=["tell"])
    irc = mocker.MagicMock()
    irc.getCallback.return_value = cb
    msg = _fake_msg(mocker)

    out = lb.dispatch(irc, msg, plugin="Later", command="tell", arg_string="alice hi")
    assert out == {"error": "denied: write commands disabled"}
    cb._callCommand.assert_not_called()


def test_dispatch_gate_check_runs_after_command_existence_check(mocker):
    """An unknown command must still surface as 'unknown command', not
    'denied: write commands disabled' — order matters for clear errors."""
    from llm import limnoria_bridge as lb

    # canonical=note, but the leaf doesn't exist on the plugin.
    cb = _stub_callback(mocker, "Note", canonical="note", commands=["search"])
    irc = mocker.MagicMock()
    irc.getCallback.return_value = cb
    msg = _fake_msg(mocker)

    # 'send' IS in MUTATING_COMMANDS, but it's not a valid command on this
    # particular cb (isCommandMethod returns False). Existence wins.
    out = lb.dispatch(irc, msg, plugin="Note", command="send", arg_string="bob hi")
    assert out == {"error": "unknown command: Note.send"}


def test_dispatch_gate_check_runs_before_capability_check(mocker):
    """A capability-blocked mutating command must surface as 'denied: write
    commands disabled' (the gate), not 'not permitted' — we don't want to
    leak which mutating commands the user would otherwise be allowed to run."""
    from llm import limnoria_bridge as lb

    cb = _stub_callback(mocker, "Later", canonical="later", commands=["tell"])
    irc = mocker.MagicMock()
    irc.getCallback.return_value = cb
    msg = _fake_msg(mocker)
    # Capability check would block — but the gate fires first.
    cap = mocker.patch.object(lb.callbacks, "checkCommandCapability", return_value="anti.cap")

    out = lb.dispatch(irc, msg, plugin="Later", command="tell", arg_string="alice hi")
    assert out == {"error": "denied: write commands disabled"}
    cap.assert_not_called()


def _make_cmd(plugin, command, arg_syntax="", description=""):
    from llm import limnoria_bridge as lb

    return lb.BridgeCommand(
        plugin=plugin, command=command, arg_syntax=arg_syntax, description=description
    )


def test_search_commands_returns_empty_for_blank_query():
    from llm import limnoria_bridge as lb

    cmds = [_make_cmd("Misc", "ping", description="Replies pong.")]
    assert lb.search_commands(cmds, "") == []
    assert lb.search_commands(cmds, "   ") == []


def test_search_commands_matches_description_text():
    """The whole point: apropos can't do this — search_commands can."""
    from llm import limnoria_bridge as lb

    cmds = [
        _make_cmd("Misc", "ping", description="Replies pong."),
        _make_cmd("Time", "tell", description="Returns the time in a given timezone."),
    ]
    out = lb.search_commands(cmds, "timezone")
    assert [c.command for c in out] == ["tell"]


def test_search_commands_ranks_multi_field_matches_higher():
    from llm import limnoria_bridge as lb

    cmds = [
        # 'channel' appears only in description
        _make_cmd("Karma", "most", description="Top karma in this channel."),
        # 'channel' appears in command, syntax, AND description
        _make_cmd(
            "Channels",
            "channel",
            arg_syntax="<channel>",
            description="Information about the channel.",
        ),
    ]
    out = lb.search_commands(cmds, "channel")
    assert out[0].command == "channel"
    assert out[1].command == "most"


def test_search_commands_respects_limit():
    from llm import limnoria_bridge as lb

    cmds = [_make_cmd("Misc", f"cmd{i}", description="ping pong") for i in range(20)]
    out = lb.search_commands(cmds, "ping", limit=5)
    assert len(out) == 5


def test_search_commands_is_case_insensitive():
    from llm import limnoria_bridge as lb

    cmds = [_make_cmd("Misc", "ping", description="Replies PONG.")]
    out = lb.search_commands(cmds, "PoNg")
    assert [c.command for c in out] == ["ping"]


def test_search_commands_multiple_tokens_all_must_match_at_least_once():
    """Tokens score additively across fields; a command matching every
    token outranks one matching only some."""
    from llm import limnoria_bridge as lb

    cmds = [
        _make_cmd("Config", "channel", description="Channel-specific config."),
        _make_cmd("Misc", "help", description="Show help for a config item."),
        _make_cmd("Misc", "ping", description="Replies pong."),
    ]
    out = lb.search_commands(cmds, "config channel")
    # Both terms hit Config.channel; only "config" hits Misc.help; nothing for ping.
    assert out[0].command == "channel"
    assert out[1].command == "help"
    assert all(c.command != "ping" for c in out)


def _enumerable_stock_command_leaves() -> dict[str, set[str]]:
    """Map canonical_plugin_name -> set of real leaf command names, derived
    from the *installed* stock Limnoria plugins (no hand-maintained list).

    Mirrors supybot.callbacks.Commands.isCommandMethod at the class level so
    no live IRC instance is needed: a leaf is a command iff it is a function
    whose positional args equal ``commandArgs`` (['self','irc','msg','args'])
    and whose name is already canonical (lowercase). Class-level inspection is
    required because some plugins (RSS) override listCommands()/isCommandMethod()
    to read instance state set in __init__, which a live-less probe lacks.
    """
    import importlib
    import inspect
    import pkgutil

    import supybot.plugins as supyplugins
    from supybot.callbacks import canonicalName

    leaves_by_canon: dict[str, set[str]] = {}
    for mod_info in pkgutil.iter_modules(supyplugins.__path__):
        if not mod_info.ispkg:
            continue
        try:
            mod = importlib.import_module(f"supybot.plugins.{mod_info.name}")
        except Exception:
            # A stock plugin that cannot import at module-load is irrelevant to
            # the bridge (it can never be a loaded callback); skip it.
            continue
        cls = getattr(mod, "Class", None)
        if cls is None:
            continue
        command_args = getattr(cls, "commandArgs", ["self", "irc", "msg", "args"])
        leaves: set[str] = set()
        for nm in dir(cls):
            if nm != canonicalName(nm):
                continue
            attr = inspect.getattr_static(cls, nm)
            if not inspect.isfunction(attr):
                continue
            try:
                argnames = inspect.getargs(attr.__code__)[0]
            except TypeError:
                continue
            if argnames == command_args:
                leaves.add(nm)
        leaves_by_canon[canonicalName(cls.__name__)] = leaves
    return leaves_by_canon


def test_mutating_commands_have_no_orphaned_entries():
    """Drift guard: every (plugin, leaf) in MUTATING_COMMANDS must correspond
    to a real, enumerable command on an installed stock Limnoria plugin.

    If a stock-plugin upgrade renames/removes a command, or an entry is typo'd,
    the orphaned tuple silently stops gating anything (the command it meant to
    classify as mutating would be exposed as read-only). This fails CI before
    that can ship. See enumerate_commands()/dispatch() membership tests on
    MUTATING_COMMANDS in limnoria_bridge.py.
    """
    from llm import limnoria_bridge as lb

    real = _enumerable_stock_command_leaves()
    orphans = sorted(
        (plugin, leaf)
        for (plugin, leaf) in lb.MUTATING_COMMANDS
        if leaf not in real.get(plugin, set())
    )
    assert not orphans, (
        "MUTATING_COMMANDS entries no longer match any enumerable stock command "
        f"(renamed/removed/typo): {orphans}"
    )


def test_drift_check_detects_orphan():
    """Meta-guard: prove the drift check above is not vacuous — a deliberately
    orphaned entry must be flagged. Protects against the helper silently
    returning empty sets (which would make the real test always pass)."""
    from llm import limnoria_bridge as lb

    real = _enumerable_stock_command_leaves()
    poisoned = (lb.MUTATING_COMMANDS - {("misc", "tell")}) | {("misc", "tel")}
    orphans = {(plugin, leaf) for (plugin, leaf) in poisoned if leaf not in real.get(plugin, set())}
    assert ("misc", "tel") in orphans
    # And the real (non-typo) entry must NOT be flagged, proving the helper
    # actually populated misc's real leaves.
    assert ("misc", "tell") not in orphans


def test_deny_commands_have_no_orphaned_entries():
    """Drift guard: every (plugin, leaf) in DENY_COMMANDS must correspond to a
    real, enumerable command on an installed stock Limnoria plugin.

    DENY_COMMANDS carries the security-critical unconditional blocks (web.fetch
    SSRF, utilities.apply/let arbitrary command re-dispatch, misc.more/
    clearmores). If a stock-plugin upgrade renames one of these — or an entry is
    typo'd — the orphaned tuple silently stops denying anything and the command
    becomes callable through the bridge. This fails CI before that can ship.
    Shares the _enumerable_stock_command_leaves() helper that
    test_drift_check_detects_orphan proves is non-vacuous.
    """
    from llm import limnoria_bridge as lb

    real = _enumerable_stock_command_leaves()
    orphans = sorted(
        (plugin, leaf) for (plugin, leaf) in lb.DENY_COMMANDS if leaf not in real.get(plugin, set())
    )
    assert not orphans, (
        "DENY_COMMANDS entries no longer match any enumerable stock command "
        f"(renamed/removed/typo): {orphans}"
    )
