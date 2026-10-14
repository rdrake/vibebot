"""Limnoria → LLM tool bridge.

Exposes loaded Limnoria plugin commands to the LLM as a single
``run_limnoria_command`` tool. Enforces a layered denial model:

1. Hard-coded ``DENY_PLUGINS`` / ``DENY_COMMANDS`` (this module).
2. Operator-set ``bridgeAllowedPlugins`` (per-channel registry).
3. ``MUTATING_COMMANDS`` gated behind ``bridgeAllowMutating``.
4. Limnoria's own capability system via ``checkCommandCapability``.

See docs/plans/archive/2026-05-02-limnoria-tool-bridge-plan.md for the full design.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from supybot import callbacks
from supybot import log as supylog

_log = supylog.getPluginLogger("LLM.bridge")

# Plugin names matched against ``cb.name()`` (the user-facing CamelCase form).
DENY_PLUGINS: frozenset[str] = frozenset(
    {
        # Auth / management — capability checks already gate non-owners, but we
        # deny at the bridge layer too so the LLM never sees these as options.
        "LLM",
        "Owner",
        "Admin",
        "Config",
        "Channel",
        "User",
    }
)

# Phase 2 Task 2: curated default allowlist used when ``bridgeAllowedPlugins``
# is empty. CamelCase to match ``cb.name()`` (the user-facing form). Each
# plugin is either pure-read or has its writes gated by ``MUTATING_COMMANDS``
# behind ``bridgeAllowMutating`` (Phase 2 Task 1). Operators can override
# with any non-empty list to opt out of the curated set; ``bridgeEnabled
# False`` disables the bridge entirely.
#
# Why a fallback constant instead of a registry default: Limnoria persists
# every registered value to disk (``_wasSet`` is True from init), so a code
# default change does not propagate to existing operators on upgrade — the
# on-disk value (the old empty default) would override the new code default.
# Treating empty-from-registry as "use the curated set" inside the bridge
# itself sidesteps that and keeps fresh installs and upgrades aligned.
DEFAULT_ALLOWED_PLUGINS: frozenset[str] = frozenset(
    {
        "Misc",
        "Time",
        "Math",
        "Utilities",
        "Seen",
        "Web",
        "Later",
        "Note",
        "Karma",
        "QuoteGrabs",
        "RSS",
        "DDG",
        # whois/whowas/latency/uptime — the "who is X / is the bot lagging"
        # questions users and IRCops ask in chat. Connection management and
        # the re-dispatch leaves are in DENY_COMMANDS.
        "Network",
    }
)

# (canonical_plugin_name, leaf_command) tuples. Both lowercase — matched
# against ``cb.canonicalName()`` (already lowercase) and the leaf name from
# ``cb.listCommands()``.
DENY_COMMANDS: frozenset[tuple[str, str]] = frozenset(
    {
        # Pastebin/scrollback — interactive only, no value via LLM.
        ("misc", "more"),
        ("misc", "clearmores"),
        # SSRF vector: arbitrary URL fetch with the bot's network privileges.
        # Capability gating answers "may this user run it," not "is this URL
        # safe for the bot's network context." Deny unconditionally.
        ("web", "fetch"),
        # Same SSRF shape as web.fetch — every one of these stock Web reads
        # calls utils.web.getUrlFd/getUrl on the caller-supplied URL with the
        # bot's network identity. They are READS, so bridgeAllowMutating does
        # NOT gate them, and "Web" is in DEFAULT_ALLOWED_PLUGINS, so they are
        # reachable in the locked-down default. web.location is the worst: it
        # skips _checkURLWhitelist entirely and follows redirects (a redirect-
        # to-internal SSRF primitive); the others honour urlWhitelist but it
        # defaults to [] which _checkURLWhitelist treats as "allow all". Deny
        # all of them at the bridge layer, exactly like web.fetch.
        ("web", "location"),
        ("web", "headers"),
        ("web", "doctype"),
        ("web", "size"),
        ("web", "title"),
        # ``apply <command> <args>`` re-dispatches through Limnoria's command
        # engine, which would bypass our per-command deny entries.
        ("utilities", "apply"),
        # ``let <name> <command> <args>`` is the same arbitrary-command-
        # redispatch shape as ``apply`` (Utilities/plugin.py:156-178); it
        # tokenizes user-supplied text and runs it via ``self.Proxy(...)``,
        # bypassing every DENY_PLUGINS / DENY_COMMANDS / MUTATING_COMMANDS
        # filter the bridge applies to first-level dispatch.
        ("utilities", "let"),
        # Network — connection management is owner-gated already, but the
        # LLM must never see it as an option; ``command``/``cmdall`` are the
        # same arbitrary-redispatch shape as ``apply``/``let`` (they run
        # ``self.Proxy(otherIrc, msg, commandAndArgs)``); ``authenticate``
        # re-runs SASL on a live connection.
        ("network", "connect"),
        ("network", "disconnect"),
        ("network", "reconnect"),
        ("network", "command"),
        ("network", "cmdall"),
        ("network", "authenticate"),
    }
)

# (canonical_plugin_name, leaf_command) tuples for commands that modify
# persistent state, send IRC traffic to a different user, or read-with-side-
# effect (e.g. marking notes as read). Both elements lowercase — matched
# against ``cb.canonicalName()`` and the leaf name from ``cb.listCommands()``.
#
# Gated by the ``bridgeAllowMutating`` channel registry value: when False
# (the default), ``enumerate_commands`` skips these and ``dispatch`` rejects
# them defense-in-depth.
#
# Sourcing: each entry is keyed to a method in a stock Limnoria plugin
# under .venv/lib/python3.14/site-packages/supybot/plugins/<Plugin>/plugin.py
# — see docs/plans/archive/2026-05-02-limnoria-bridge-task-1-implementation-plan.md
# for line-level citations.
MUTATING_COMMANDS: frozenset[tuple[str, str]] = frozenset(
    {
        # Misc — sends a private message to a third user.
        ("misc", "tell"),
        ("misc", "noticetell"),
        # Later — offline-tell DB.
        ("later", "tell"),
        ("later", "remove"),
        ("later", "undo"),
        # Note — registered-user notes DB. ``note``/``next`` call
        # ``db.setRead`` but the side effect is a read-receipt scoped to
        # the caller's own notes (the plugin enforces
        # ``note.frm/note.to == user.id``); classified read-only.
        ("note", "send"),
        ("note", "reply"),
        ("note", "unsend"),
        # Karma — clear/dump/load all touch persistent state.
        ("karma", "clear"),
        ("karma", "dump"),
        ("karma", "load"),
        # QuoteGrabs — grab/ungrab insert/delete rows.
        ("quotegrabs", "grab"),
        ("quotegrabs", "ungrab"),
        # RSS — add/remove register/unregister feeds. ``rss`` and
        # ``info`` are reads on their face but ``update_feed_if_needed``
        # (RSS/plugin.py:396) can call ``announce_feed`` (line 434),
        # which queues PRIVMSG/NOTICE to every channel subscribed to the
        # feed (line 553-557). Classified mutating to keep LLM-triggered
        # reads from pushing entries into third-party channels.
        # NB: nested ``announce add/remove/list/channels`` leaves are
        # NOT classified here — see "Ambiguous classifications" #3 in
        # the plan; multi-word leaves are out of scope for Task 1.
        ("rss", "add"),
        ("rss", "remove"),
        ("rss", "rss"),
        ("rss", "info"),
        # Forward-look: not in Phase 2 Task 2's default allowlist but
        # classified now so the gate is correct when they're added.
        # Quote — ChannelIdDatabasePlugin write commands plus the
        # plugin-local ``replace`` override.
        ("quote", "add"),
        ("quote", "remove"),
        ("quote", "change"),
        ("quote", "replace"),
        # Todo — user-scoped todo DB writes.
        ("todo", "add"),
        ("todo", "remove"),
        ("todo", "setpriority"),
        ("todo", "change"),
        # Factoids — channel-scoped fact DB writes. ``whatis`` looks
        # like a read but ``_replyFactoids`` calls ``_updateRank``
        # (Factoids/plugin.py:372-383, 397, 420) which UPDATEs
        # ``relations.usage_count`` whenever ``keepRankInfo`` is True.
        # ``keepRankInfo`` defaults True (Factoids/config.py:85-87) so
        # by default ``whatis`` writes the DB on every call.
        ("factoids", "learn"),
        ("factoids", "alias"),
        ("factoids", "lock"),
        ("factoids", "unlock"),
        ("factoids", "forget"),
        ("factoids", "change"),
        ("factoids", "whatis"),
        # Scheduler — every leaf except ``list`` is a write.
        ("scheduler", "add"),
        ("scheduler", "remind"),
        ("scheduler", "remove"),
        ("scheduler", "repeat"),
    }
)


class BufferingIrcProxy(callbacks.ReplyIrcProxy):
    """An ``IrcProxy`` that captures replies into a list instead of
    queueing them onto the IRC connection.

    All the rich-reply machinery (``reply``, ``error``, ``replies``,
    ``replySuccess``, etc.) flows through ``reply()`` and ``error()``
    in the base class, so overriding those two is sufficient.
    """

    def __init__(self, irc, msg):
        super().__init__(irc, msg)
        self.buffer: list[str] = []

    def reply(self, s, msg=None, **kwargs):  # noqa: ARG002 (signature compat)
        self.buffer.append(s)
        return None

    def error(self, s, msg=None, **kwargs):  # noqa: ARG002 (signature compat)
        # Buffer the text first so the dispatch error envelope can include
        # it on the Raise=True path (the exception is caught in dispatch()).
        self.buffer.append(s)
        if kwargs.get("Raise"):
            # Preserve ReplyIrcProxy.error()'s control-flow contract — some
            # commands use Raise=True for early-exit. See callbacks.py:675.
            raise callbacks.Error(s)
        return None


@dataclass(frozen=True)
class BridgeCommand:
    """One enumerated, callable Limnoria command."""

    plugin: str  # cb.name() — CamelCase, used in operator config
    command: str  # leaf command name from cb.listCommands()
    arg_syntax: str  # first line of method.__doc__
    description: str  # remaining lines of method.__doc__, joined


def enumerate_commands(
    irc: Any,
    msg: Any,
    allowed_plugins: frozenset[str],
    *,
    allow_mutating: bool = False,
) -> Iterator[BridgeCommand]:
    """Yield every loaded command the LLM is allowed to call.

    A command is yielded when ALL of:
    - Its plugin is in ``allowed_plugins`` (operator allowlist).
    - Its plugin is NOT in ``DENY_PLUGINS`` (hard deny).
    - Its (canonical_plugin, leaf) tuple is NOT in ``DENY_COMMANDS``.
    - When ``allow_mutating`` is False (the default), its
      (canonical_plugin, leaf) tuple is NOT in ``MUTATING_COMMANDS``.
    - ``checkCommandCapability(msg, cb, leaf)`` returns falsy
      (i.e. allowed for the calling user).

    The capability check uses the string form of the leaf name to
    mirror ``_callCommand``'s pattern at supybot/callbacks.py:1591;
    list form ``[leaf]`` triggers an AssertionError because the leaf
    is not the plugin's canonical name.
    """
    for cb in irc.callbacks:
        plugin_name = cb.name()
        if plugin_name in DENY_PLUGINS:
            continue
        if plugin_name not in allowed_plugins:
            continue
        canonical = cb.canonicalName()
        for leaf in cb.listCommands():
            if (canonical, leaf) in DENY_COMMANDS:
                continue
            if not allow_mutating and (canonical, leaf) in MUTATING_COMMANDS:
                continue
            denial = callbacks.checkCommandCapability(msg, cb, leaf)
            if denial:
                continue
            try:
                method = cb.getCommandMethod([leaf])
            except Exception:
                # Nested command groups (e.g. RSS's 'announce add') surface as
                # multi-word leaves whose list form is not the plugin's
                # canonical name, so getCommandMethod raises an AssertionError.
                # Skip the leaf rather than let one plugin's command shape abort
                # the whole enumeration — and with it the entire @ask request,
                # since a plugin with nested groups can be in the allowlist.
                _log.debug("bridge: skipping uninspectable command %s.%s", plugin_name, leaf)
                continue
            doc_lines = (method.__doc__ or "").strip().splitlines()
            arg_syntax = doc_lines[0].strip() if doc_lines else ""
            description = " ".join(line.strip() for line in doc_lines[1:]).strip()
            yield BridgeCommand(
                plugin=plugin_name,
                command=leaf,
                arg_syntax=arg_syntax,
                description=description,
            )


def search_commands(
    commands: list[BridgeCommand],
    query: str,
    *,
    limit: int = 10,
) -> list[BridgeCommand]:
    """Rank ``commands`` by relevance to ``query`` and return the top ``limit``.

    Scoring is whitespace-tokenized substring matching across
    ``plugin.command``, ``arg_syntax``, and ``description`` — case-insensitive.
    Each token contributes one point per field it appears in, so a match in
    every field outranks a match in just one. Designed to compensate for
    Limnoria's ``Misc.apropos``, which only matches command names.

    Returns at most ``limit`` results in descending score order. Ties keep
    the source ordering (``enumerate_commands``'s plugin/leaf order).
    """
    tokens = [t for t in query.lower().split() if t]
    if not tokens:
        return []
    scored: list[tuple[int, int, BridgeCommand]] = []
    for idx, c in enumerate(commands):
        name = f"{c.plugin}.{c.command}".lower()
        syntax = c.arg_syntax.lower()
        desc = c.description.lower()
        score = 0
        for tok in tokens:
            if tok in name:
                score += 1
            if tok in syntax:
                score += 1
            if tok in desc:
                score += 1
        if score:
            scored.append((-score, idx, c))
    scored.sort()
    return [c for _score, _idx, c in scored[:limit]]


def dispatch(
    irc: Any,
    msg: Any,
    *,
    plugin: str,
    command: str,
    arg_string: str,
    allowed: frozenset[str] = frozenset(),
    allow_mutating: bool = False,
) -> dict[str, Any]:
    """Run ``plugin.command arg_string`` through Limnoria's command path.

    Layered checks before dispatch:
    1. Plugin must resolve via ``irc.getCallback(plugin)``.
    2. Plugin must not be in ``DENY_PLUGINS``, and its ``cb.name()`` must be
       in ``allowed`` (empty = ``DEFAULT_ALLOWED_PLUGINS``, as the handler
       resolves it for ``enumerate_commands``).
    3. (canonical_plugin, command) must not be in ``DENY_COMMANDS``.
    4. ``cb.isCommandMethod(command)`` must be True.
    5. When ``allow_mutating`` is False (the default), (canonical_plugin,
       command) must not be in ``MUTATING_COMMANDS``. Defense in depth on
       top of ``enumerate_commands``'s filter — even if the LLM hallucinates
       a write command, the dispatch path still rejects.
    6. ``checkCommandCapability(msg, cb, command)`` must be falsy.

    On success, returns ``{"status": "ok", "reply": "<captured text>"}``.
    On any check failure or uncaught exception, returns
    ``{"error": "<reason>"}``. The shape matches ``AssistantToolExecutor._ok``
    / ``_err`` (assistant.py) so the assistant loop's
    ``last_successful_tool`` guard in ``service.assistant_completion`` fires
    correctly.
    """
    # arg_string is LLM-generated and may carry secrets (e.g. a URL with an API
    # key). Never log its content at INFO — only its length. Full content goes
    # to DEBUG (off in production), mirroring the reply logging below.
    _log.info(
        "bridge call: %s.%s args=%i chars nick=%s channel=%s allow_mutating=%s",
        plugin,
        command,
        len(arg_string),
        getattr(msg, "nick", "?"),
        getattr(msg, "channel", "?"),
        allow_mutating,
    )
    _log.debug("bridge call args: %s.%s %r", plugin, command, arg_string)
    cb = irc.getCallback(plugin)
    if cb is None:
        _log.info("bridge result: %s.%s -> error: unknown plugin", plugin, command)
        return {"error": f"unknown plugin: {plugin}"}
    if cb.name() in DENY_PLUGINS or cb.name() not in (allowed or DEFAULT_ALLOWED_PLUGINS):
        _log.info("bridge result: %s.%s -> error: denied (plugin)", plugin, command)
        return {"error": f"denied: {plugin}.{command}"}
    if (cb.canonicalName(), command) in DENY_COMMANDS:
        _log.info("bridge result: %s.%s -> error: denied (command)", plugin, command)
        return {"error": f"denied: {plugin}.{command}"}
    if not cb.isCommandMethod(command):
        _log.info("bridge result: %s.%s -> error: unknown command", plugin, command)
        return {"error": f"unknown command: {plugin}.{command}"}
    if not allow_mutating and (cb.canonicalName(), command) in MUTATING_COMMANDS:
        _log.info(
            "bridge result: %s.%s -> error: denied (mutation gate closed)",
            plugin,
            command,
        )
        return {"error": "denied: write commands disabled"}
    denial = callbacks.checkCommandCapability(msg, cb, command)
    if denial:
        _log.info("bridge result: %s.%s -> error: not permitted", plugin, command)
        return {"error": f"not permitted: {plugin}.{command}"}

    proxy = BufferingIrcProxy(irc, msg)
    # Arg tokenization errors are user/LLM input errors (malformed bracket/pipe/
    # quote syntax, callbacks.py:431). Surface the specific message so the model
    # can correct its call — these carry no internal detail.
    try:
        tokens = callbacks.tokenize(arg_string, channel=msg.channel, network=irc.network)
    except Exception as exc:  # noqa: BLE001 — translating to JSON envelope
        _log.info("bridge result: %s.%s -> arg error: %s", plugin, command, exc)
        return {"error": str(exc) or exc.__class__.__name__}

    # Plugin execution errors may carry internal detail (DB hosts, internal
    # addresses, stack hints). Log the full exception internally but return a
    # generic message to the model. Preserve any output the plugin buffered
    # before raising so partial results aren't silently dropped.
    try:
        # Positional args; keyword `args=tokens` would land in **kwargs and
        # break wrap()-based commands. See callbacks.py:1213.
        cb._callCommand([command], proxy, msg, tokens)
    except Exception as exc:  # noqa: BLE001 — translating to JSON envelope
        _log.info("bridge result: %s.%s -> exception: %s", plugin, command, exc)
        envelope: dict[str, Any] = {"error": "command failed"}
        partial = "\n".join(proxy.buffer)
        if partial:
            envelope["partial_output"] = partial
        return envelope
    reply = "\n".join(proxy.buffer)
    _log.debug("bridge result: %s.%s -> ok reply=%r", plugin, command, reply)
    _log.info("bridge result: %s.%s -> ok (%i chars)", plugin, command, len(reply))
    return {"status": "ok", "reply": reply}
