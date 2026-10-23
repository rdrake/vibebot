"""MkDocs hook: render ```irc fences as a styled IRC transcript.

A fence body is IRC as a client shows it:

    <you>     VibeBot, any aurora tonight?
    <VibeBot> Unlikely overhead, but up to 35% toward the
              northern horizon.
    * you prods VibeBot
    -!- carol has joined #lounge

An indented line continues the message above it; a blank line leaves a gap.
Nick colours match the landing page: the bot is cyan, `you` orange, and
anyone else gets a stable colour hashed from the nick. A nick named inside
a line (the bot's, or anyone who speaks in the fence) takes its colour too,
as a client colours mentions; `you` doesn't, since it's also just a word.
"""

from __future__ import annotations

import re
import zlib
from html import escape

_BOT = "vibebot"
_OTHER_COLOURS = ("c-carol", "c-dave", "c-alice")
_MSG = re.compile(r"<([^>\s]+)>\s?(.*)")
_ACTION = re.compile(r"\*\s+(\S+)\s?(.*)")
_EVENT = re.compile(r"(-!-|-->|<--)\s?(.*)")
_COMMAND = re.compile(r"^(@\w+)")


def _colour(nick: str) -> str:
    low = nick.lower()
    if low == _BOT:
        return "c-bot"
    if low == "you":
        return "c-alice"
    return _OTHER_COLOURS[zlib.crc32(low.encode()) % len(_OTHER_COLOURS)]


def _mentions(entries: list[list[str]]) -> re.Pattern[str]:
    """Match the bot's nick and each speaker's where a line names them."""
    nicks = {_BOT} | {nick.lower() for kind, nick, _ in entries if kind in ("msg", "act")}
    nicks.discard("you")
    names = "|".join(re.escape(n) for n in sorted(nicks, key=len, reverse=True))
    return re.compile(rf"(?<![\w@])({names})(?!\w)", re.IGNORECASE)


def _text(raw: str, mentions: re.Pattern[str]) -> str:
    html = _COMMAND.sub(r'<span class="k">\1</span>', escape(raw.strip()))
    return mentions.sub(lambda m: f'<em class="{_colour(m[1])}">{m[1]}</em>', html)


def _lines(source: str) -> list[list[str]]:
    """Group the fence into [kind, nick, text] entries."""
    out: list[list[str]] = []
    for line in source.splitlines():
        if not line.strip():
            if out and out[-1][0] != "gap":
                out.append(["gap", "", ""])
        elif line[0].isspace() and out and out[-1][0] in ("msg", "act"):
            out[-1][2] += " " + line.strip()
        elif m := _MSG.fullmatch(line):
            out.append(["msg", m[1], m[2]])
        elif m := _ACTION.fullmatch(line):
            out.append(["act", m[1], m[2]])
        elif m := _EVENT.fullmatch(line):
            out.append(["evt", m[1], m[2]])
        else:
            out.append(["raw", "", line])
    while out and out[-1][0] == "gap":
        out.pop()
    return out


def format_irc(source, language, css_class, options, md, **kwargs):
    items = []
    entries = _lines(source)
    mentions = _mentions(entries)
    for kind, nick, text in entries:
        if kind == "gap":
            items.append('<li class="is-gap" aria-hidden="true"></li>')
        elif kind == "msg":
            items.append(
                f'<li><b class="{_colour(nick)}">{escape(nick)}</b> <span>{_text(text, mentions)}</span></li>'
            )
        elif kind == "act":
            items.append(
                f'<li class="is-act"><b>*</b> <span><em class="{_colour(nick)}">'
                f"{escape(nick)}</em> {_text(text, mentions)}</span></li>"
            )
        elif kind == "evt":
            items.append(
                f'<li class="is-evt"><b>{escape(nick)}</b> <span>{_text(text, mentions)}</span></li>'
            )
        else:
            items.append(f'<li class="is-raw"><b></b> <span>{_text(text, mentions)}</span></li>')
    return f'<div class="{css_class}" role="log"><ol>{"".join(items)}</ol></div>'


def on_config(config, **kwargs):
    superfences = config.mdx_configs.setdefault("pymdownx.superfences", {})
    superfences.setdefault("custom_fences", []).append(
        {"name": "irc", "class": "vb-irc", "format": format_irc}
    )
    return config
