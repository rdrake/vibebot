"""LLM plugin implementation."""

from __future__ import annotations

import collections
import contextlib
import json
import logging
import mimetypes
import re
import secrets
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple
from urllib.parse import urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import supybot.callbacks as callbacks
import supybot.conf as conf
import supybot.httpserver as httpserver
import supybot.ircdb as ircdb
import supybot.ircmsgs as ircmsgs
import supybot.ircutils as ircutils
import supybot.log as log
import supybot.schedule as schedule
from supybot import world
from supybot.commands import getopts, optional, wrap
from supybot.i18n import PluginInternationalization

from . import (
    agenda,
    apikeys,
    ircquery,
    limnoria_bridge,
    meme,
    meme_render,
    meme_topics,
    spaceweather,
    statuspage,
)
from .assistant import PENDING_TASK_TOOLS
from .context import ContextConfig, ConversationContext, Role
from .executor import LLMExecutor, RecursiveSubmitError
from .identity import UNAUTHENTICATED_KEY_PREFIX, Caller, Identity
from .persistence import LLMDatabase, ReminderRow, ScheduledLlmTaskRow
from .profile import (
    PROFILE_CHAT,
    PROFILE_DRAW,
    PROFILE_REMIND_ACTION,
    PROFILE_VERSE,
    PROFILES,
)
from .service import (
    SILENT_TOOLS,
    AssistantRequestContext,
    AssistantResult,
    ImageResult,
    LLMService,
    account_from_server_tags,
    exponential_backoff,
    irc_has_caps,
    truncate_to_byte_budget,
    truncate_to_word_boundary,
    validate_external_url,
)
from .tracing import TraceFilter, generate_request_id, request_id
from .usertz import (
    SOURCE_CTCP,
    SOURCE_SET,
    UTC_DEFAULT,
    UserTz,
    needs_zone,
    offset_from_ctcp_time,
    parse_zone,
)
from .verse import reactions
from .verse.aging import AgingOutcome
from .verse.avatar import (
    _VerseToolResult,
    build_story_world_context,
    build_verse_context_block,
    build_verse_system_prompt,
    dispatch_verse_edit,
    is_ooc,
    make_verse_denial_handlers,
    make_verse_extra_handlers,
    make_verse_tool_specs,
    strip_ooc,
)
from .verse.compaction import CompactionOutcome
from .verse.store import VerseStore, db_path_for_channel

if TYPE_CHECKING:
    from PIL import Image
    from supybot.ircmsgs import IrcMsg

    from .assistant import ToolCallbackResult, ToolResult
    from .service import MemePicture

_ = PluginInternationalization("LLM")

# Icon shown when Google grounding/search was used in the response
GROUNDING_ICON = "\U0001f310"  # 🌐 (globe with meridians)

# Icon shown when the delivered image came from a prompt the safety rewriter
# changed. The picture is still the subject that was asked for -- the rewriter
# is held to that -- but the wording around it moved, so a picture that reads
# slightly off is explained rather than mysterious.
REWORDED_ICON = "\U0001f501"  # 🔁 (clockwise arrows)

# Commands that support long-term memory extraction
_MEMORY_COMMANDS = frozenset({"ask"})

# Usage-row label for image spend. Namespaced like "compaction:compact" rather
# than reusing "draw", because a turn writes BOTH a text row and an image row
# and counting draws by row would otherwise double. One image generated, one
# row: `WHERE command = 'draw:image'` is the image bill.
_IMAGE_USAGE_COMMAND = "draw:image"

# C0 control characters except TAB (\x09), LF (\x0a), CR (\x0d).
# Includes ESC (\x1b) which starts ANSI sequences like \x1b[6n whose
# brackets crash Limnoria's nested-command tokenizer.
_CTRL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# CTCP ACTION frame (a "/me" line). The \x01 delimiters live inside the
# range _CTRL_CHAR_RE scrubs, so both inFilter and doPrivmsg unwrap the
# payload explicitly rather than letting the scrub flatten "/me waves"
# into the literal text "ACTION waves".
_ACTION_RE = re.compile(r"^\x01ACTION\s+(.*)\x01$", re.DOTALL)

# Characters that may sit between a bot-nick address and the rest of the
# line. Same set NickInMiddle uses, so the two matchers agree on what a
# word boundary is.
_ADDRESS_SEPARATORS = " \t,:;"
_NICK_TOKEN_RE = re.compile(r"[^\s,;:]+")

# Selector name for a status page, shared by statusPageUrls and
# statusQueryablePages. ASCII-only, so plain .lower() is sufficient for
# case-insensitive uniqueness.
_STATUS_PAGE_NAME_RE = re.compile(r"\A[A-Za-z0-9._-]{1,32}\Z")

_REQUEST_CONTEXT_CAPABILITIES = frozenset(
    {
        "llm.ask",
        "llm.draw",
        "llm.verse",
        "llm.verse.gm",
        "llm.verse.edit",
        "owner",
        "admin",
        "trusted",
    }
)

# Pending-task mutation tools whose successful execution already produced
# a user-visible emoji reaction. When the assistant loop ends with one of
# these as the last successful tool AND no follow-up text, the chat
# reply is suppressed to avoid a duplicate ack. See Task B5 of the
# 2026-04-30 reminder simplification plan.
#
# The suppressed turn still needs a stand-in in stored context: an empty
# assistant turn reads back as an unanswered request, so the next turn
# repeats the mutation. On 2026-09-03 "remind me to do the needful in 60s"
# followed by "do so kindly" set the same reminder twice for that reason.
_REMINDER_MUTATION_CONTEXT_NOTES = {
    "set_reminder": "reminder set",
    "cancel_pending_task": "pending task cancelled",
    "cancel_all_pending_tasks": "all pending tasks cancelled",
}
_REMINDER_MUTATION_TOOLS = frozenset(_REMINDER_MUTATION_CONTEXT_NOTES)

_FULL_ANSWER_LABEL = "Full answer"

# Entry routes for "just talk" ambient messages (nick-addressed text and bare
# `vibebot foo`), as opposed to explicit commands. Sticky @rp roleplay (Slice 3)
# promotes only these to roleplay turns — explicit @ask / @rp stay as-is.
_AMBIENT_ENTRY_ROUTES = frozenset({"addressed", "invalid_command"})

# Drop a second identical addressed message to the same channel within this many
# seconds. Kills relay echoes (LarryBot re-injecting a "vibebot …" line) and any
# double entry-point that would otherwise spawn a duplicate answer — or, worse, a
# duplicate illustrated story. Short enough that a user genuinely re-asking after
# a beat is not blocked.
_DISPATCH_DEDUP_WINDOW = 12.0

# How many of the bot's own msgids to remember for +draft/reply addressing.
# A busy channel sees a few hundred bot lines a day; replies to anything
# older than that are rare enough to need the nick.
_OWN_MSGID_CAP = 2000

# How many (channel, nick) -> last msgid entries the react tool remembers.
# One per recent speaker per channel; anyone quieter than that is too stale
# to react to anyway.
_LAST_MSGID_CAP = 2000


def _parse_iso_utc(tag: Any) -> datetime | None:
    """An ISO-8601 string as an aware UTC datetime; ``None`` if unreadable."""
    if not isinstance(tag, str):
        return None
    try:
        ts = datetime.fromisoformat(tag)
    except ValueError:
        return None
    return ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts.astimezone(UTC)


# Longest emoji the react tool accepts, in code points. ZWJ sequences
# (👨‍👩‍👧, 🏳️‍🌈) take several; anything longer is text, not a reaction.
_REACT_EMOJI_MAX_CHARS = 8

# delete_last_reply: how many of the bot's own lines to remember per
# channel, and how far apart two lines can be and still be one reply (a
# long answer goes out as several lines over a second or two).
_OWN_LINES_PER_TARGET = 20
_REPLY_BURST_SECONDS = 5.0

# notify_when_online limits: MONITOR holds 128 nicks per connection on
# AfterNET; one person should not fill it, and a watch nobody saw fire in a
# week is stale.
_WATCHES_PER_REQUESTER = 5
_WATCH_MAX_AGE = 7 * 86400

# WHOX query token for irc_lookup kind=who. Limnoria's own join-time WHO
# uses "1" (irclib do354); a different token keeps the two from mixing.
_WHO_LOOKUP_TOKEN = "7"

# The stored react-only note ("[reacted 👍 to bob's message]"), as the model
# reproduces it in plain text; see _dispatch_assistant_reply.
_FAKE_REACT_NOTE_RE = re.compile(r"\s*\[(?:reacted|no recent message|deleted my)\b", re.IGNORECASE)

# Re-ask the network about an unflagged nick at most this often. A WHO per
# stranger is cheap, one per stranger per hour is nothing, and the answer is
# stable — nobody stops being a bot mid-afternoon.
_BOT_PROBE_INTERVAL = 3600.0

# Explicit image-intent cues: an ambient verse-mention asking for a single
# PICTURE, so it draws an image instead of a story. ("illustrate" is deliberately
# excluded here — that reads as an illustrated STORY, see _ILLUSTRATE_INTENT_RE.)
_DRAW_INTENT_RE = re.compile(
    r"\b(draw|sketch|paint|render|picture of|image of|a pic of|drawing of)\b",
    re.IGNORECASE,
)

# Explicit request for an ILLUSTRATED tale — opens the generous multi-image
# storybook rather than the prose-first ambient default.
_ILLUSTRATE_INTENT_RE = re.compile(
    r"\b(illustrat\w*|comic|storybook|picture\s*book|with\s+(?:pictures|pics|art|illustrations))\b",
    re.IGNORECASE,
)

# A message shaped like a question (interrogative opener, or ends in "?"). Used
# on the chat path to spot a genuine question so a real-world one — asked in a
# channel whose overlay says "tell the tallest tales, go mental" — gets a
# straight, true answer instead of invented lore. Deliberately broad on the
# opener; the canon check (verse_context is None) is what keeps in-world
# questions ("what have the stinky lads done") out of the factual path.
_QUESTION_SHAPE_RE = re.compile(
    r"^\s*(?:who|whos|what|whats|when|where|which|whose|whom|why|how|"
    r"is|are|was|were|do|does|did|can|could|would|will|should|has|have|had)\b",
    re.IGNORECASE,
)

# Chat-overlay replacement for a real-world question with no canon reference.
# Swapped in for the channel's "tallest tales / go mental" overlay so vibebot
# answers the actual question (school milk → Margaret Thatcher) instead of
# fictionalising it. The carve-out lets it still play along when the question is
# genuinely about the channel's own running characters (which usually trip the
# canon check anyway, but this covers the odd un-tracked in-joke).
_FACTUAL_CHAT_OVERLAY = (
    "The user is asking a real question. Answer it directly, accurately, and "
    "truthfully from real-world knowledge — the actual answer, concise and "
    "correct. Do NOT fictionalise it, invent facts, or spin it into a tall tale. "
    "Only if the question is genuinely about this channel's own running "
    "characters or in-jokes may you answer in that playful spirit. Stay vibebot "
    "— never claim to be a user from the channel."
)

# Slice 2 canon layer: appended to the chat-path canon block when
# verseChatRecordEnabled + an opted-in avatar. Invites the model to persist a
# genuinely new durable fact to canon, sparingly, so canon grows from ordinary
# chat and not only @rp roleplay. Kept terse to avoid over-eager recording.
_VERSE_CHAT_RECORD_NUDGE = (
    "If this exchange establishes a genuinely NEW, durable fact about these "
    "characters or the world (not small talk, opinions, or one-off banter), you "
    "may call verse_record ONCE to save it to canon. When in doubt, don't."
)

# Appended to the canon block on the CHAT answer path (not the draw path): a
# grounded reply should have the world's flavour, not read as a dry fact recital
# — this is what the "livelier grounded answer" is asking for.
_VERSE_CHAT_FLAVOUR_NUDGE = (
    "Answer in this world's voice — vivid and a little larger-than-life — not a "
    "flat recital of these facts."
)


class ReminderScheduleResult(NamedTuple):
    """Result of scheduling a reminder."""

    ok: bool
    message: str
    # What is now pending, in fixed wording: "<message> in <n>s". The
    # confirmation above is model-authored and often names neither, which
    # leaves a later "make it 30s" with no antecedent.
    summary: str = ""
    # The parser asked the user something instead of scheduling; ``message``
    # is the question. Nothing was set, so no ⏰ — and no ❌ either.
    question: bool = False


class PreflightResult(NamedTuple):
    """Result of the shared command preflight check.

    ``blocked`` is True when the command should not proceed (the preflight
    already sent the appropriate error reply and logged usage). ``caller``
    is the Caller resolved for this message; everything after preflight
    reads it rather than ``msg``.
    """

    blocked: bool
    caller: Caller


class VerseRoute(NamedTuple):
    """Result of _verse_route_for.  Populated when the message should be
    handled by the verse engine (verseEnabled, llm.verse capability, avatar
    exists, not OOC)."""

    avatar_id: int
    system_prompt: str
    tools: list[dict]
    store: VerseStore


@dataclass(frozen=True)
class CommandInfo:
    """Metadata for a user-facing command, used to generate help."""

    name: str
    args: str
    description: str
    examples: tuple[str, ...]
    category: str  # "generation", "memory", "utility"


COMMAND_REGISTRY: tuple[CommandInfo, ...] = (
    CommandInfo(
        name="ask",
        args="<question>",
        description=(
            "Ask the AI a question. Supports conversation context "
            "(follow-up questions) and vision (include image URLs)."
        ),
        examples=(
            "@ask What is the capital of France?",
            "@ask Describe this: https://example.com/image.jpg",
            "@ask And what about Germany?  (follow-up using context)",
        ),
        category="generation",
    ),
    CommandInfo(
        name="draw",
        args="<prompt>",
        description="Generate an image from a text description.",
        examples=(
            "@draw A sunset over mountains in watercolor style",
            "@draw A cyberpunk cityscape at night",
        ),
        category="generation",
    ),
    CommandInfo(
        name="meme",
        args="[--font <name>] [--top] <template or image URL> | <caption> [| <caption> ...]",
        description=(
            "Caption one of about a thousand meme templates, or any image by URL. "
            "Name the template, then the captions, separated by |. "
            "'list <word>' finds templates."
        ),
        examples=(
            "@meme drake | left on unread | left on read",
            "@meme distracted boyfriend | me | a new side project | my actual job",
            "@meme list cat",
        ),
        category="generation",
    ),
    CommandInfo(
        name="story",
        args="<brief>",
        description=(
            "Generate an illustrated page (prose plus a few AI illustrations) and "
            "post a link when it's ready. Tells an illustrated tale OR explains a "
            "concept with diagrams as a learning aid. No verse mode needed."
        ),
        examples=(
            "@story an illustrated tale of stinky lads winning the pub quiz",
            "@story explain how photosynthesis works, with diagrams",
        ),
        category="generation",
    ),
    CommandInfo(
        name="forget",
        args="[channel]",
        description=(
            "Clear your volatile memory (conversation context) "
            "for the current or specified channel."
        ),
        examples=("@forget", "@forget #channel"),
        category="memory",
    ),
    CommandInfo(
        name="memories",
        args="[del <id> | edit <id> <text> | clear | cleanup]",
        description=(
            "Manage your non-volatile memory (stored facts the bot "
            "remembers about you across conversations)."
        ),
        examples=(
            "@memories",
            "@memories delete 3",
            "@memories edit 5 corrected fact",
            "@memories clear",
        ),
        category="memory",
    ),
    CommandInfo(
        name="instruct",
        args="[<instruction> | clear]",
        description=(
            "Set persistent instructions that shape how @ask responds to you. "
            "Your instruction is prepended to the system prompt."
        ),
        examples=(
            "@instruct You are Captain Picard. Respond in character.",
            "@instruct Respond only in haiku",
            "@instruct clear",
            "@instruct",
        ),
        category="memory",
    ),
    CommandInfo(
        name="avatar",
        args="[<persona> | clear]",
        description=(
            "Set the persona that shapes your avatar in verse-enabled channels. "
            "Independent of @instruct — this only affects the verse, not @ask."
        ),
        examples=(
            "@avatar A moss-covered tree spirit who speaks in riddles.",
            "@avatar clear",
            "@avatar",
        ),
        category="memory",
    ),
    CommandInfo(
        name="remind",
        args="[<text> | list | del <id> | clear | admin <list|del|clear> <nick> [<id>...]]",
        description="Set and manage reminders using natural language.",
        examples=(
            "@remind in 30 minutes check the build",
            "@remind list",
            "@remind delete abc1",
            "@remind clear",
            "@remind admin list someone",
            "@remind admin del someone abc1",
            "@remind admin clear someone",
        ),
        category="utility",
    ),
    CommandInfo(
        name="tz",
        args="[<zone> | clear]",
        description=(
            "Set your time zone so a reminder for 9am fires at 9am your time. "
            "Use an IANA name like America/Toronto. Unset, reminders go by your "
            "client's clock if it answers CTCP TIME, else UTC."
        ),
        examples=("@tz America/Toronto", "@tz Europe/London", "@tz clear", "@tz"),
        category="utility",
    ),
    CommandInfo(
        name="usage",
        args="[nick | #channel]",
        description="Show API usage statistics.",
        examples=("@usage", "@usage someone", "@usage #channel"),
        category="utility",
    ),
    CommandInfo(
        name="verseopt",
        args="<in|out>",
        description=(
            "Opt your avatar in or out of the verse for this channel. "
            "Requires the llm.verse capability."
        ),
        examples=("@verseopt in", "@verseopt out"),
        category="utility",
    ),
    CommandInfo(
        name="verse",
        args="",
        description=(
            "Show your current scene one-liner in the verse. "
            "Requires the llm.verse capability and a verse-enabled channel."
        ),
        examples=("@verse",),
        category="utility",
    ),
    CommandInfo(
        name="rp",
        args="<text>",
        description=(
            "Roleplay: take one in-character verse turn as your avatar (persona, "
            "scene, narrative tools). A bare canon mention only grounds a normal "
            "reply now — use @rp to opt into roleplay. Requires the llm.verse "
            "capability and a verse-enabled channel."
        ),
        examples=("@rp Archie kicks the door open and bellows for the lads",),
        category="utility",
    ),
    CommandInfo(
        name="look",
        args="[<target>]",
        description=(
            "Show your current scene, or describe a named entity in the verse. "
            "Requires the llm.verse capability and a verse-enabled channel."
        ),
        examples=("@look", "@look The Clearing", "@look alice"),
        category="utility",
    ),
    CommandInfo(
        name="who",
        args="",
        description=(
            "List active avatars and their locations in the verse. "
            "Requires the llm.verse capability and a verse-enabled channel."
        ),
        examples=("@who",),
        category="utility",
    ),
    CommandInfo(
        name="versedump",
        args="[#channel] [--format=json]",
        description=(
            "Dump the full verse state for a channel as JSON. Requires the llm.verse.gm capability."
        ),
        examples=("@versedump", "@versedump #afnet", "@versedump #afnet --format=json"),
        category="utility",
    ),
    CommandInfo(
        name="versepurge",
        args="[#channel] [token]",
        description=(
            "Wipe the verse store for a channel. "
            "First call issues a confirmation token; second call with the token performs the wipe. "
            "Requires the llm.verse.gm capability."
        ),
        examples=("@versepurge #afnet", "@versepurge #afnet a1b2c3"),
        category="utility",
    ),
    CommandInfo(
        name="versecompact",
        args="<channel>",
        description=(
            "Manually run retention compaction for a channel: summarise old "
            "events into a single digest entry. Requires the llm.verse.gm capability."
        ),
        examples=("@versecompact #afnet",),
        category="utility",
    ),
    CommandInfo(
        name="versedit",
        args="<verb> <args...> [#channel]",
        description=(
            "Edit the verse universe: add/pin/unpin/set/name/desc/retire/restore/"
            "relate/unrelate/event/editevent/delevent/show. "
            "Requires the llm.verse.edit capability."
        ),
        examples=(
            "@versedit add npc Assgas Archie :: Y11 windbag",
            "@versedit pin Assgas Archie",
            "@versedit retire #42",
        ),
        category="utility",
    ),
    CommandInfo(
        name="canon",
        args="<lock|unlock|forget> <name>",
        description=(
            "Lock or release a character as durable canon (always remembered, "
            "aging-exempt). 'forget' is an alias for 'unlock'. "
            "Requires the llm.verse.edit capability and a verse-enabled channel."
        ),
        examples=("@canon lock Harry", "@canon unlock Harry", "@canon forget Harry"),
        category="utility",
    ),
    CommandInfo(
        name="channels",
        args="[--min <n>] [<channel or glob>]",
        description=(
            "List the busiest public channels on this network with user counts "
            "and topics. A channel name shows that channel; a glob like #linux* "
            "filters; --min hides channels below n users."
        ),
        examples=("@channels", "@channels #linux", "@channels --min 5 #linux*"),
        category="utility",
    ),
    CommandInfo(
        name="model",
        args="[<channel>]",
        description=(
            "Show which model answered the bot's last reply in a channel and how "
            "long ago, next to the chat model the channel is set to use."
        ),
        examples=("@model", "@model #tv"),
        category="utility",
    ),
    CommandInfo(
        name="names",
        args="[<channel>]",
        description=(
            "List who is in a channel at this moment, including channels the bot has "
            "not joined. Defaults to the current channel."
        ),
        examples=("@names", "@names #linux"),
        category="utility",
    ),
)


class LLMHTTPCallback(httpserver.SupyHTTPServerCallback):
    """HTTP callback to serve LLM-generated files (images, code)."""

    name = "LLM"
    public = True

    def __init__(self, plugin: LLM) -> None:
        """Initialize with plugin reference."""
        super().__init__()
        self._plugin = plugin

    def _get_web_dir(self) -> str:
        """Get the web directory for LLM files."""
        http_root = self._plugin.registryValue("httpRoot")
        if http_root:
            return http_root
        return conf.supybot.directories.data.web.dirize("llm")

    def doGet(self, handler: httpserver.RequestHandler, path: str) -> None:  # noqa: N802
        """Serve static files from LLM web directory."""
        # Remove leading slash
        path = path.lstrip("/")

        # No index page — help docs are on GitHub Pages
        if path == "":
            handler.send_response(404)
            handler.end_headers()
            return

        # Security: prevent directory traversal (early check before path operations)
        if ".." in path or path.startswith("/"):
            handler.send_response(403)
            handler.end_headers()
            return

        web_dir = Path(self._get_web_dir())
        filepath = web_dir / path

        # Security: resolve symlinks and verify path is under web root
        try:
            resolved_web_dir = web_dir.resolve()
            resolved_filepath = filepath.resolve()

            # Ensure resolved path is under web directory (Python 3.9+)
            if not resolved_filepath.is_relative_to(resolved_web_dir):
                handler.send_response(403)
                handler.end_headers()
                return
        except (OSError, ValueError):
            handler.send_response(403)
            handler.end_headers()
            return

        # Check file exists
        if not resolved_filepath.is_file():
            handler.send_response(404)
            handler.end_headers()
            return

        # Determine content type
        content_type, _ = mimetypes.guess_type(str(resolved_filepath))
        if content_type is None:
            content_type = "application/octet-stream"

        try:
            with open(resolved_filepath, "rb") as f:
                content = f.read()

            handler.send_response(200)
            handler.send_header("Content-Type", content_type)
            handler.send_header("Content-Length", str(len(content)))
            handler.end_headers()
            handler.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError):
            # Client disconnected - this is normal, ignore silently
            pass
        except OSError:
            try:
                handler.send_response(500)
                handler.end_headers()
            except (BrokenPipeError, ConnectionResetError):
                pass


def _safe_database_path(configured: str, default: str) -> str:
    """Return the operator-configured DB path, or ``default`` when it is empty
    or contains a ``..`` traversal component.

    ``databasePath`` is operator-set (capability-gated), so this is
    defense-in-depth: it refuses an obvious traversal like
    ``../../../etc/passwd`` rather than handing it straight to SQLite. Plain
    absolute paths are the normal case and are kept as-is.
    """
    if not configured:
        return default
    if ".." in Path(configured).parts:
        return default
    return configured


def _patch_irc_dojoin(plugin: LLM) -> None:
    """Replace supybot.irclib.Irc.doJoin to skip slow auto-queries on JOIN.

    Why: Limnoria's stock doJoin queues MODE +b (ban-list) and a WHO sync on
    every channel join. Nothing in this codebase reads ban state, and on
    servers with account-tag + extended-join the WHO is redundant — both
    queries serialize behind connection registration and meaningfully delay
    startup notification on rejoin.

    The patch always drops MODE +b and conditionally drops the WHO (gated by
    :meth:`LLM._will_skip_auto_who`). The plain MODE <channel> query is kept
    because Limnoria reads channel-mode state in many places.

    Re-patches on every plugin __init__ so the closure tracks the current
    LLM instance after a reload. Cheap; the patch is global to all Irc
    instances, so a multi-instance LLM plugin would have the last-init
    instance win — not a real concern for this single-plugin deployment.
    """
    from supybot import irclib, ircmsgs

    def doJoin(self, msg):  # noqa: N802
        if msg.nick != self.nick:
            return
        channel = msg.args[0]
        skip_who = plugin._will_skip_auto_who(self)
        if not skip_who:
            self.queueMsg(ircmsgs.who(channel, args=("%tuhnairf,1",)))
            # Track start of WHO sync so do315 can compute elapsed time.
            self.startedSync[channel] = time.time()
        self.queueMsg(ircmsgs.mode(channel))  # plain channel modes; ends with 329
        # Always skip MODE +b — nothing in the codebase reads ban-list state.
        # If WHO is skipped, do NOT touch startedSync — do315 will never arrive
        # and the dict would leak across rejoins.

    irclib.Irc.doJoin = doJoin


def _patch_irc_docapnew() -> None:
    """Make doCapNew also request experimental caps (draft/multiline et al).

    Why: Limnoria's stock doCapNew filters CAP NEW announcements through
    REQUEST_CAPABILITIES only, ignoring REQUEST_EXPERIMENTAL_CAPABILITIES
    even when experimentalExtensions is enabled. AfterNET's bouncer
    advertises draft/multiline via CAP NEW post-SASL, so without this
    patch Limnoria never requests it and long replies fall back to
    @more pagination.
    """
    from supybot import conf, irclib

    def doCapNew(self, msg):  # noqa: N802
        if len(msg.args) != 3:
            log.warning("Bad CAP NEW from server: %r", msg)
            return
        caps = msg.args[2].split()
        assert caps, "Empty list of capabilities"
        self._addCapabilities(msg.args[2], msg)
        if self.state.fsm.state == irclib.IrcStateFsm.States.SHUTTING_DOWN:
            return
        want = irclib.Irc.REQUEST_CAPABILITIES
        if conf.supybot.protocols.irc.experimentalExtensions():
            want = want | irclib.Irc.REQUEST_EXPERIMENTAL_CAPABILITIES
        new = set(self.state.capabilities_ls) & want - self.state.capabilities_ack
        if new:
            self.requestCapabilities(new)

    irclib.Irc.doCapNew = doCapNew


# Draft caps irc_lookup kind=history and delete_last_reply need. Limnoria's
# list has neither, and AfterNET (Nefarious) gates CHATHISTORY and REDACT on
# them. Requested like draft/multiline: only with experimentalExtensions.
_EXTRA_EXPERIMENTAL_CAPS = frozenset({"draft/chathistory", "draft/message-redaction"})


def _patch_request_caps() -> None:
    """Add _EXTRA_EXPERIMENTAL_CAPS to what Limnoria asks for at connect."""
    from supybot import irclib

    irclib.Irc.REQUEST_EXPERIMENTAL_CAPABILITIES = (
        set(irclib.Irc.REQUEST_EXPERIMENTAL_CAPABILITIES) | _EXTRA_EXPERIMENTAL_CAPS
    )


def _format_compaction_outcome(
    co: CompactionOutcome,
    ao: AgingOutcome | None,
    *,
    min_keep_events: int,
) -> str:
    """Render a single human-readable line covering both compaction and
    (optional) aging counts.

    The two halves used to be separate INFO log records — operators had
    to correlate them by channel name. Folding them into one message
    keeps a daily-pass channel summary on a single line.

    ``ao=None`` is for callers (``@versecompact``) that compact one
    channel without running aging; the aged clause is omitted.
    """
    if co.state == "compacted":
        head = f"compacted {co.total_events} events"
    elif co.state == "skipped_below_floor":
        head = f"skipped (only {co.total_events} events; floor is {min_keep_events})"
    elif co.state == "skipped_no_events":
        head = f"skipped (no events past retention; total {co.total_events})"
    elif co.state == "skipped_disabled":
        head = "skipped (retention disabled)"
    else:  # forward-compat: unknown future state strings
        head = co.state
    if ao is None:
        return head
    aged_kept = ao.scanned - ao.retired
    return f"{head}; aged {ao.retired} entities (kept {aged_kept})"


def _format_ago(seconds: float) -> str:
    """Compact age for @model: 45s, 12m, 3h, 2d."""
    seconds = max(0, int(seconds))
    for size, unit in ((86400, "d"), (3600, "h"), (60, "m")):
        if seconds >= size:
            return f"{seconds // size}{unit}"
    return f"{seconds}s"


class LLM(callbacks.Plugin):
    """AI-powered commands backed by Anthropic and xAI models.

    Provides ask, code, draw commands with multi-provider support.
    """

    threaded = True  # Commands run in threads for non-blocking I/O

    def __init__(self, irc: callbacks.Irc) -> None:
        """Initialize plugin.

        Args:
            irc: IRC connection instance
        """
        super().__init__(irc)
        self.llm_service = LLMService(self)
        self.log = log.getPluginLogger("LLM")
        self.log.addFilter(TraceFilter())

        installed = apikeys.install_secret_filter()
        self.log.info(
            # %s only, never %d: supybot routes log args through
            # utils.str.format, whose mini-language has no %d. A %d is left
            # literal and the positional args shift into the %s slots, so this
            # line rendered as "…%d handler(s)… covered: 4" in production.
            "secret redaction: %s handler(s) filtered, %s variable(s) covered: %s",
            installed,
            len(apikeys.secret_var_names()),
            ", ".join(apikeys.secret_var_names()) or "none",
        )

        # Apply configured log level to plugin and service loggers
        self._apply_log_level()

        # Global concurrency cap for all LLM I/O. See
        # docs/plans/2026-05-06-async-llm-concurrency.md
        self._llm_executor = LLMExecutor(
            max_concurrency=self.registryValue("maxConcurrentLLMCalls"),
            log=self.log,
        )

        self.startup_time = time.time()  # Track startup for ZNC playback filtering
        self.build_info = self._get_build_info()

        # Initialize database for persistence (before context, which loads from DB)
        configured_db_path = self.registryValue("databasePath")
        default_db_path = str(Path(conf.supybot.directories.data()) / "LLM.db")
        db_path = _safe_database_path(configured_db_path, default_db_path)
        if configured_db_path and db_path != configured_db_path:
            self.log.warning(
                "databasePath %r rejected (path traversal); using default %r",
                configured_db_path,
                db_path,
            )
        self.db = LLMDatabase(db_path)

        _patch_irc_dojoin(self)
        _patch_irc_docapnew()
        _patch_request_caps()

        # Initialize conversation context (loads persisted conversations from DB)
        self._init_context()

        # In-memory per-command rate-limit buckets: "{command}:{account}" -> deque of timestamps
        self._rate_buckets: dict[str, collections.deque[float]] = {}
        self._rate_buckets_lock = threading.Lock()

        # Bot-loop guard. "{network}:{nick}" -> is the nick +B, and when did we
        # last ask. The network knows who is a robot (AfterNET advertises
        # BOT=B) but only says so in the WHOX status field, which arrives on
        # request — prod skips the WHO on join, so we ask about a nick the
        # first time it talks to us. Both in-memory: a reload re-asks.
        self._bot_flags: dict[str, bool] = {}
        self._bot_probed_at: dict[str, float] = {}
        # (target, bot nick) -> (consecutive replies, last reply time).
        self._bot_reply_counts: dict[tuple[str, str], tuple[int, float]] = {}
        self._bot_loop_lock = threading.Lock()
        # msgids of the bot's own lines, learned from echo-message, so an
        # IRCv3 +draft/reply to one of them counts as addressing the bot.
        # Bounded LRU keyed (network, msgid); see _remember_own_msgid.
        self._own_msgids: collections.OrderedDict[tuple[str, str], None] = collections.OrderedDict()
        self._own_msgids_lock = threading.Lock()
        # (network, channel, nick) -> msgid of that nick's latest line in the
        # channel, for the react tool. Bounded LRU; see _remember_last_msgid.
        self._last_msgids: collections.OrderedDict[tuple[str, str, str], str] = (
            collections.OrderedDict()
        )
        self._last_msgids_lock = threading.Lock()
        # Entries changed since the last _flush_last_msgids, with seen time.
        self._last_msgids_dirty: dict[tuple[str, str, str], tuple[str, float]] = {}
        self._load_last_msgids()
        # (network, channel) -> (model, time) of the last assistant reply, for
        # @model. In memory only: a restart forgets it, which @model says.
        self._served_models: dict[tuple[str, str], tuple[str, float]] = {}
        # Pending LIST / NAMES queries: fed by do322/do353 et al. on the
        # driver thread, awaited by @channels/@names and the irc_lookup tool.
        self._irc_queries = ircquery.IrcQueryRegistry()

        # Channels already warned about an empty verseModel (warn once per channel).
        self._verse_model_warned: set[str] = set()

        # Recent addressed dispatches for duplicate suppression:
        # (channel, normalized-text) -> last-seen epoch. Drops relay echoes and
        # double entry-points within _DISPATCH_DEDUP_WINDOW (see
        # _is_duplicate_dispatch). In-memory; lost on reload, by design.
        self._recent_dispatch: dict[tuple[str, str], float] = {}
        self._recent_dispatch_lock = threading.Lock()

        # Sticky @rp roleplay sessions: (channel, account-or-nick) -> expiry epoch
        # (inf = never auto-expire). While present+unexpired, the ambient path
        # treats the caller's plain messages as roleplay turns. Slice 3.
        self._roleplay_sticky: dict[tuple[str, str], float] = {}
        self._roleplay_sticky_lock = threading.Lock()

        # Reminders and scheduled LLM tasks: the rows are the schedule, and
        # one wakeup timer fires whatever is due (see agenda.py).
        self.agenda = agenda.Agenda(self.db, owner_tz=self._owner_tz)
        self._agenda_wakeup_at: float | None = None
        self._agenda_wakeup_lock = threading.Lock()

        # CTCP TIME fallback for users with no @tz; answers (and silences)
        # persist in user_clock_offsets. Waiters hold the in-flight probe,
        # keyed by lowercased nick, that the NOTICE fills in.
        self._ctcp_tz_lock = threading.Lock()
        self._ctcp_tz_waiters: dict[str, tuple[threading.Event, list[str]]] = {}

        # Serializes worker-thread irc.queueMsg calls (see _safe_queue).
        self._irc_send_lock = threading.Lock()

        # Recency-attributed verse reaction signal (see verse/reactions.py).
        # Last verse line the bot said per (network, channel); read by doTagmsg.
        self._last_bot_line: dict[tuple[str, str], dict] = {}
        self._reaction_log_lock = threading.Lock()

        # Per-channel VerseStore cache (keyed by channel name).
        self._verse_stores: dict[str, VerseStore] = {}
        self._verse_stores_lock = threading.Lock()

        # In-memory versepurge confirmation tokens: channel -> (token, expires_at).
        # Resets on plugin reload/bot restart (by design; documented in operator guide).
        self._versepurge_tokens: dict[str, tuple[str, float]] = {}
        self._versepurge_tokens_lock = threading.Lock()

        # Arm the agenda wakeup from the persisted rows. Overdue reminders
        # and tasks come due on its first tick, never inside __init__.
        self._arm_agenda()

        # Startup notification tracking
        self._pending_channels: set[str] = set()
        self._startup_notified: bool = False

        # Only register HTTP callback if using Limnoria's built-in web directory
        # (i.e., httpRoot is not configured). When httpRoot is set, an external
        # web server (e.g., nginx) is expected to serve files from that path.
        if not self.registryValue("httpRoot"):
            self._http_callback = LLMHTTPCallback(self)
            httpserver.hook("llm", self._http_callback)
        else:
            self._http_callback = None

        # Schedule periodic file cleanup (runs every hour)
        # Defensive: remove any existing event first (handles plugin reloads)
        with contextlib.suppress(KeyError):
            schedule.removeEvent("llm_file_cleanup")

        schedule.addPeriodicEvent(
            self._run_file_cleanup,
            3600,  # 1 hour in seconds
            name="llm_file_cleanup",
            now=False,  # Don't run immediately on startup
        )

        with contextlib.suppress(KeyError):
            schedule.removeEvent("llm_last_msgids_flush")
        schedule.addPeriodicEvent(
            self._flush_last_msgids, 60, name="llm_last_msgids_flush", now=False
        )

        # Safety poll for pending tasks (5-minute fallback for event-driven wakeups).
        # Initialize the in-flight gate BEFORE scheduling — a synchronous fire
        # during construction would NameError without the flag.
        self._safety_poll_inflight = threading.Event()
        with contextlib.suppress(KeyError):
            schedule.removeEvent("llm_pending_tasks")

        schedule.addPeriodicEvent(
            self._enqueue_safety_poll,
            self._SAFETY_POLL_INTERVAL,
            name="llm_pending_tasks",
            now=False,
        )

        # Status page poller. Lifecycle state is advanced here and nowhere
        # else; the tool's inline fetch writes only _status_read_cache.
        self._status_poll_inflight = threading.Event()
        # Every field is keyed by canonical source id (statuspage.canonical_source).
        # The ownership split from 2026-08-09 is unchanged and load-bearing:
        # _status_state is advanced by the poller ONLY, so a user asking "is it
        # down?" cannot consume an announcement. The tool's inline fetch writes
        # _status_read_cache and _status_last_fetch.
        self._status_state: dict[str, statuspage.StatusState] = {}
        self._status_read_cache: dict[str, statuspage.Snapshot] = {}
        self._status_last_fetch: dict[str, float] = {}
        self._status_announce_times: list[float] = []
        self._status_history_cache: dict[str, tuple[statuspage.HistoryEntry, ...]] = {}
        self._status_history_at: dict[str, float] = {}
        self._status_history_failed_at: dict[str, float] = {}
        # Queryable-only pages: lazily filled, never polled, never announced,
        # and never granted lifecycle state.
        self._status_query_cache: dict[str, statuspage.Snapshot] = {}
        self._status_query_failed_at: dict[str, float] = {}
        self._status_cursor: str | None = None
        # Aurora feed caches, shared by the tool and the storm announcer.
        # _sw_lock guards both dicts and is never held across a network fetch.
        self._sw_lock = threading.Lock()
        self._sw_cache: dict[str, tuple[datetime, Any]] = {}
        self._sw_places: dict[str, tuple[float, spaceweather.Place | None]] = {}
        with contextlib.suppress(KeyError):
            schedule.removeEvent("llm_status_poll")
        self._schedule_status_poll()

        # Storm announcer. _sw_state is advanced by the poller only, one pass
        # at a time (_sw_poll_inflight), so it needs no lock.
        self._sw_state = spaceweather.SwState()
        self._sw_poll_inflight = threading.Event()
        self._sw_place_warned: set[str] = set()
        with contextlib.suppress(KeyError):
            schedule.removeEvent("llm_spaceweather_poll")
        self._schedule_spaceweather_poll()

        # Event-driven queue wakeup state
        self._next_wakeup_time: float | None = None
        self._queue_wakeup_lock = threading.Lock()
        self._schedule_queue_wakeup()  # rebuild from DB on startup

        # Register callback for live log level changes
        conf.supybot.plugins.LLM.logLevel.addCallback(self._on_log_level_change)

        # Daily verse compaction timer (PR 3 / E3). Registry keys
        # ``verseCompactionDailyAt`` and ``verseCompactionMinKeepEvents``
        # are added by F1; until then ``_register_compaction_timer``'s
        # try/except falls back to documented defaults so the timer
        # still arms on un-configured installs.
        self._compaction_timer_name = "llm_verse_compact"
        self._register_compaction_timer()

    def _apply_log_level(self) -> None:
        """Set plugin logger levels from the logLevel config value."""
        level_name = self.registryValue("logLevel")
        level = getattr(logging, level_name, logging.WARNING)
        self.log.setLevel(level)
        self.llm_service.log.setLevel(level)

    def _on_log_level_change(self, *args: object) -> None:
        """Called when logLevel config changes at runtime."""
        self._apply_log_level()

    def die(self) -> None:
        """Clean up when plugin is unloaded."""
        # Shutdown the executor before mutating shared state so workers
        # see closing=True at their commit points. Brief drain gives
        # already-running workers a chance to flush final
        # db.log_usage / queueMsg writes before we close the DB.
        if hasattr(self, "_llm_executor"):
            self._llm_executor.shutdown()
            self._llm_executor.drain(timeout=2.0)

        # Stop the shared +typing keepalive thread. It cannot send
        # +typing=done on the way out — _safe_queue drops sends once shutdown
        # has begun — and it does not need to: clients expire the state.
        if hasattr(self, "llm_service"):
            self.llm_service.typing.stop()

        # Clean up expired reminders from database
        if hasattr(self, "db"):
            if hasattr(self, "_last_msgids_dirty"):
                self._flush_last_msgids()
            # Close the main-thread DB connection. Worker-thread thread-local
            # connections are released as those threads exit; we don't track
            # them centrally, so sqlite ResourceWarnings can still appear
            # under reload-heavy/test-heavy workloads (see pyproject.toml).
            self.db.close()

        # Remove scheduled cleanup event
        with contextlib.suppress(KeyError):
            schedule.removeEvent("llm_file_cleanup")
        with contextlib.suppress(KeyError):
            schedule.removeEvent("llm_pending_tasks")
        with contextlib.suppress(KeyError):
            schedule.removeEvent("llm_last_msgids_flush")
        with contextlib.suppress(KeyError):
            schedule.removeEvent("llm_status_poll")
        with contextlib.suppress(KeyError):
            schedule.removeEvent("llm_spaceweather_poll")
        with contextlib.suppress(KeyError):
            schedule.removeEvent("llm_startup_check")
        with contextlib.suppress(KeyError):
            schedule.removeEvent("llm_queue_wakeup")
        with contextlib.suppress(KeyError):
            schedule.removeEvent(self._AGENDA_WAKEUP_EVENT)
        # Daily compaction timer teardown (PR 3 / E3).
        if hasattr(self, "_compaction_timer_name"):
            self._cancel_compaction_timer()

        # Only unhook HTTP callback if we registered
        if self._http_callback is not None:
            httpserver.unhook("llm")
        super().die()

    def _run_file_cleanup(self) -> None:
        """Scheduled cleanup of old generated files."""
        try:
            self.llm_service.run_scheduled_cleanup()
            self.log.debug("Scheduled file cleanup completed")
        except Exception as e:
            self.log.error("Scheduled file cleanup failed: %s", e)

    def _enqueue_safety_poll(self) -> None:
        """Submit a single safety-poll worker, deduped by ``_safety_poll_inflight``.

        The flag prevents overlapping polls when a long LLM dispatch holds
        the worker; we never queue more than one inflight at a time.
        """
        if self._llm_executor.closing:
            return
        if self._safety_poll_inflight.is_set():
            return
        self._safety_poll_inflight.set()
        try:
            fut = self._llm_executor.submit("safety_poll", self._check_pending_tasks)
        except Exception:
            # Synchronous submit failure must not leave the flag stuck set.
            self._safety_poll_inflight.clear()
            raise
        fut.add_done_callback(lambda _f: self._safety_poll_inflight.clear())

    def _schedule_queue_wakeup(self, at_time: float | None = None) -> None:
        """Schedule a one-shot wakeup for the next due queue task.

        If *at_time* is given it is used directly; otherwise the earliest
        ``next_attempt_at`` is queried from the database.  A wakeup is only
        scheduled when it would fire earlier than any existing one.

        Args:
            at_time: Optional explicit wakeup timestamp.  When provided the
                database is not queried.
        """
        # Called from the safety-poll worker and from request threads; the
        # check -> removeEvent -> addEvent sequence must not interleave.
        with self._queue_wakeup_lock:
            self._schedule_queue_wakeup_locked(at_time)

    def _schedule_queue_wakeup_locked(self, at_time: float | None) -> None:
        if at_time is None:
            at_time = self.db.get_next_due_time()
        if not isinstance(at_time, (int, float)):
            return

        now = time.time()

        # Clamp past-due timestamps to now + 1 so Limnoria doesn't discard them
        effective = max(at_time, now + 1)

        # Skip if an existing wakeup is already earlier and still in the future
        if (
            self._next_wakeup_time is not None
            and self._next_wakeup_time <= effective
            and self._next_wakeup_time > now
        ):
            return

        # Replace any existing wakeup
        with contextlib.suppress(KeyError):
            schedule.removeEvent("llm_queue_wakeup")

        schedule.addEvent(
            self._enqueue_safety_poll,
            effective,
            name="llm_queue_wakeup",
        )
        self._next_wakeup_time = effective

    def _check_pending_tasks(self) -> None:
        """Poll pending tasks and deliver completed/failed/expired results.

        Each result is delivered independently so that one delivery failure
        does not cascade to the rest of the batch.
        """
        try:
            # The wakeup that triggered this call has fired; clear it so
            # _schedule_queue_wakeup can schedule the next one.
            self._next_wakeup_time = None

            # Build set of channels the bot is currently in
            deliverable_channels: set[str] = set()
            for irc_conn in list(world.ircs):
                # IrcDict.keys() is a Python-level loop a JOIN/PART on the
                # driver thread can break; list() of items() is one C call.
                # Not list(channels): that yields the lowercased keys.
                deliverable_channels.update(k for k, _v in list(irc_conn.state.channels.items()))

            results = self.llm_service.check_pending_tasks(deliverable_channels)

            for r in results:
                try:
                    self._deliver_pending_result(r)
                except Exception as e:
                    self.log.error(
                        "Delivery failed for task_id=%s nick=%s: %s",
                        r.task_id,
                        r.nick,
                        e,
                    )

            # Schedule the next wakeup based on remaining queue state
            if not self._llm_executor.closing:
                self._schedule_queue_wakeup()
        except Exception as e:
            self.log.error("Pending task check failed: %s", e)

    def _status_now(self) -> float:
        """Indirection point so tests can pin the clock."""
        return time.time()

    def _status_monotonic(self) -> float:
        """Monotonic clock for deadlines. Separate indirection from _status_now
        so tests can pin them independently — and so a wall-clock adjustment
        cannot corrupt a deadline."""
        return time.monotonic()

    def _status_parse_pages(self, raw_entries, *, key: str, cap: int, warn: bool) -> dict[str, str]:
        """Parse ``Name=url`` (or bare ``url``) entries into name -> source.

        One grammar, one parser, both config keys. Bad entries are dropped
        rather than raising: one typo must not disable the feature.

        Split on the FIRST '=' — names forbid '=' and, being in a
        space-separated list, spaces, so this is unambiguous.
        """
        pages: dict[str, str] = {}
        lowered: set[str] = set()
        for raw in raw_entries or []:
            text = str(raw).strip()
            if not text:
                continue
            name, sep, url = text.partition("=")
            bare = not sep
            if bare:
                name, url = "", text
            source = statuspage.canonical_source(url)
            if source is None:
                if warn:
                    self.log.warning("Ignoring unusable %s entry: %s", key, text[:100])
                continue
            if bare:
                # A genuine bare url (no '=' at all) falls back to its host.
                # An explicit but empty name ("=url") is a typo, not a bare
                # url, and must not silently claim that page's name.
                #
                # The derived name is sanitised to always satisfy
                # _STATUS_PAGE_NAME_RE rather than checked against it: a host
                # is operator-chosen but not operator-typed, so a host that is
                # merely long (>32 chars, e.g. some AWS regional status
                # hosts) or carries a port must still be polled. The regex
                # stays a hard gate only for explicit Name= entries below,
                # where a rejection is a visible typo the operator can fix.
                name = self._status_bare_name(source)
            elif not _STATUS_PAGE_NAME_RE.match(name):
                if warn:
                    self.log.warning("Ignoring %s entry with an unusable name: %s", key, text[:100])
                continue
            if name.lower() in lowered:
                if warn:
                    self.log.warning("Ignoring duplicate %s name: %s", key, name)
                continue
            if source in pages.values():
                # Two names for one page: the later one would look valid in
                # @config while never appearing in the enum.
                existing = next(n for n, s in pages.items() if s == source)
                if warn:
                    self.log.warning("Ignoring %s entry %s: same page as %s", key, name, existing)
                continue
            pages[name] = source
            lowered.add(name.lower())
        if len(pages) > cap:
            if warn:
                self.log.warning(
                    "%s lists %i usable pages; using the first %i", key, len(pages), cap
                )
            pages = dict(list(pages.items())[:cap])
        return pages

    def _status_polled_pages(self, *, warn: bool = True) -> dict[str, str]:
        return self._status_parse_pages(
            self.registryValue("statusPageUrls"),
            key="statusPageUrls",
            cap=self._STATUS_MAX_SOURCES,
            warn=warn,
        )

    def _status_named_pages(self, *, warn: bool = True) -> dict[str, str]:
        """Name -> canonical source for every configured page, polled first.

        The single source of truth for the tool's ``service`` enum, for
        resolving that argument, and for the prune sets. Purely a function of
        config: nothing here may come from a fetched payload, or a page could
        rename itself into another page's selector.
        """
        pages = self._status_polled_pages(warn=warn)
        polled_sources = set(pages.values())
        lowered = {n.lower() for n in pages}
        for name, source in self._status_parse_pages(
            self.registryValue("statusQueryablePages"),
            key="statusQueryablePages",
            cap=self._STATUS_MAX_QUERYABLE,
            warn=warn,
        ).items():
            if source in polled_sources or name.lower() in lowered:
                if warn:
                    self.log.warning("Ignoring queryable page %s: already configured", name)
                continue
            pages[name] = source
        return pages

    def _status_sources(self, *, warn: bool = True) -> list[str]:
        """Canonical, deduplicated, capped list of POLLED status pages.

        Order is the operator's. ``warn`` gates diagnostics and defaults to on
        for the poller's ~2-minute cadence; request-path callers pass False, or
        one typo'd entry logs once per chat message.
        """
        return list(self._status_polled_pages(warn=warn).values())

    def _status_host(self, source: str) -> str:
        """Display host for a canonical source id. Operator-derived, always safe."""
        try:
            return urlparse(source).hostname or source
        except ValueError:
            return source

    def _status_bare_name(self, source: str) -> str:
        """Selector name for a bare-URL entry: always matches _STATUS_PAGE_NAME_RE.

        The host is sanitised to the allowed charset and truncated to fit,
        with the port appended (when present) so two ports on the same host
        still get distinct names — e.g. ``x.example`` and ``x.example-8443``
        for ``https://x.example`` and ``https://x.example:8443``. A genuine
        collision (two bare URLs that land on the same derived name) is left
        for the existing duplicate-name check to drop and log; this only
        guarantees the name is well-formed, not unique.
        """
        host = self._status_host(source)
        port = ""
        try:
            p = urlparse(source).port
        except ValueError:
            p = None
        if p is not None:
            port = f"-{p}"
        safe = re.sub(r"[^A-Za-z0-9._-]", "-", host) or "page"
        budget = max(1, 32 - len(port))
        return safe[:budget] + port

    def _status_prune_sources(self, sources: list[str], queryable: list[str] | None = None) -> None:
        """Drop state for sources no longer configured.

        Two different keep-sets, deliberately, across eight structures.
        Lifecycle state is pruned against the POLLED set: a queryable page
        must never hold any. The history and query caches are pruned against
        polled UNION queryable — pruning history against the polled set alone
        deletes an allowlisted page's history (up to 4 MB, cached for an hour)
        on the very next poll, 120 seconds after it was fetched, backoff and
        all. The query cache and its failure backoff get the same treatment:
        without it, a page's query reading is evicted on the next poll rather
        than aged out by its own TTL and cap.
        """
        polled = set(sources)
        both = polled | set(queryable or ())
        for holder, keep in (
            (self._status_state, polled),
            (self._status_read_cache, polled),
            (self._status_last_fetch, polled),
            (self._status_history_cache, both),
            (self._status_history_at, both),
            (self._status_history_failed_at, both),
            (self._status_query_cache, both),
            (self._status_query_failed_at, both),
        ):
            # list(holder) snapshots the keys before iterating: a request
            # thread's _status_query_snapshot can insert into
            # _status_query_cache concurrently, and iterating the live dict
            # would raise "dictionary changed size during iteration" — which
            # _run_status_poll's outer handler swallows, losing the whole
            # poll pass including its announcements. pop(..., None) rather
            # than del: another thread's own prune-adjacent write (the query
            # evictor, or a second poll racing this one) may have already
            # removed the key.
            for stale in [k for k in list(holder) if k not in keep]:
                holder.pop(stale, None)

    def _schedule_status_poll(self) -> None:
        """Arm the next status poll as a one-shot.

        A self-rescheduling one-shot rather than addPeriodicEvent: the
        periodic wrapper re-adds itself under the same name after every
        firing, so a missing die() teardown makes the next plugin load trip
        ``assert name not in self.events`` (schedule.py:88). The one-shot also
        re-reads its interval each tick. Same pattern as _schedule_queue_wakeup.

        Arms unconditionally even when ``statusPageUrl`` is empty:
        ``_run_status_poll`` already no-ops on an empty URL, and this is the
        only re-arm site besides a poll's own done-callback. Disarming here
        would mean re-enabling the key at runtime does nothing until the
        plugin reloads.
        """
        if self._llm_executor.closing:
            return
        with contextlib.suppress(KeyError):
            schedule.removeEvent("llm_status_poll")
        schedule.addEvent(
            self._enqueue_status_poll,
            self._status_now() + self._STATUS_POLL_INTERVAL,
            name="llm_status_poll",
        )

    def _enqueue_status_poll(self) -> None:
        """Submit one status-poll worker, deduped by ``_status_poll_inflight``.

        Mirrors _enqueue_safety_poll: never more than one inflight, and the
        flag is cleared by a done-callback so a hung poll cannot wedge it.

        When a poll is already inflight this returns without re-arming: the
        inflight poll's own done-callback re-arms unconditionally, so a
        second call here would race it (``schedule.addEvent`` asserts the
        name isn't already present, outside its own lock).
        """
        if self._llm_executor.closing:
            return
        if self._status_poll_inflight.is_set():
            return
        self._status_poll_inflight.set()
        try:
            fut = self._llm_executor.submit("status_poll", self._run_status_poll)
        except Exception:
            self._status_poll_inflight.clear()
            self._schedule_status_poll()
            raise
        fut.add_done_callback(lambda _f: self._status_poll_inflight.clear())
        fut.add_done_callback(lambda _f: self._schedule_status_poll())

    # --- storm announcer -----------------------------------------------------

    def _sw_now(self) -> datetime:
        """Indirection point so tests can move the announcer's clock (aware UTC)."""
        return datetime.now(UTC)

    def _schedule_spaceweather_poll(self) -> None:
        """Arm the next storm-announcer pass as a one-shot (see _schedule_status_poll).

        Arms even when no channel announces: the pass itself returns before
        any fetch then, and turning auroraAnnounce on at runtime works
        without a reload.
        """
        if self._llm_executor.closing:
            return
        with contextlib.suppress(KeyError):
            schedule.removeEvent("llm_spaceweather_poll")
        schedule.addEvent(
            self._enqueue_spaceweather_poll,
            self._status_now() + self._SPACEWEATHER_POLL_INTERVAL,
            name="llm_spaceweather_poll",
        )

    def _enqueue_spaceweather_poll(self) -> None:
        """Submit one announcer pass, deduped by ``_sw_poll_inflight``.

        Same contract as _enqueue_status_poll: an inflight pass re-arms from
        its own done-callback, so this returns without re-arming.
        """
        if self._llm_executor.closing:
            return
        if self._sw_poll_inflight.is_set():
            return
        self._sw_poll_inflight.set()
        try:
            fut = self._llm_executor.submit("spaceweather_poll", self._run_spaceweather_poll)
        except Exception:
            self._sw_poll_inflight.clear()
            self._schedule_spaceweather_poll()
            raise
        fut.add_done_callback(lambda _f: self._sw_poll_inflight.clear())
        fut.add_done_callback(lambda _f: self._schedule_spaceweather_poll())

    def _sw_announce_channels(self) -> list[str]:
        # Copy before iterating, as _announce_status does: channel state
        # mutates under JOIN/PART on the IRC thread.
        return [
            channel
            for channel in sorted(self._all_known_channels())
            if self.registryValue("auroraAnnounce", channel)
        ]

    def _sw_channel_places(self, channel: str) -> tuple[list[tuple[str, spaceweather.Place]], bool]:
        """(label, place) for each configured place that resolves this pass,
        and whether any lookup raised.

        Labels are the configured names, never the geocoder's. A lookup
        failure skips the place for this pass only; a place that matches
        nothing is skipped (warned once).
        """
        out: list[tuple[str, spaceweather.Place]] = []
        failed = False
        for raw in self.registryValue("auroraAnnouncePlaces", channel) or []:
            label = raw.replace("_", " ").strip()
            if not label:
                continue
            try:
                place = self._sw_place(label)
            except spaceweather.FetchError as exc:
                self.log.info("spaceweather: geocode failed for %r: %s", label, exc)
                failed = True
                continue
            if place is None:
                key = label.casefold()
                if key not in self._sw_place_warned:
                    self._sw_place_warned.add(key)
                    self.log.warning(
                        "spaceweather: auroraAnnouncePlaces entry %r in %s matches no place",
                        label,
                        channel,
                    )
                continue
            out.append((label, place))
        return out, failed

    def _sw_mean_cloud(
        self, place: spaceweather.Place, start: datetime, end: datetime, *, now: datetime
    ) -> int | None:
        """Mean cloud over ``[start, end]``; None on failure or when the forecast
        ends more than an hour before ``end`` (a partial night is no figure)."""
        try:
            series = self._sw_cloud_series(place, now=now)
        except spaceweather.FetchError as exc:
            self.log.info("spaceweather: cloud cover failed for %s: %s", place.name, exc)
            return None
        if not series or series[-1][0] < end - timedelta(hours=1):
            return None
        return spaceweather.mean_cloud(spaceweather.hours_within(series, start=start, end=end))

    @staticmethod
    def _sw_dark_hours(place: spaceweather.Place, night: date) -> tuple[datetime, datetime] | None:
        """``night``'s dark window; None when it never gets dark (or bad tz)."""
        try:
            start, end = spaceweather.dark_window(place.lat, place.lon, night, place.tz)
        except (ZoneInfoNotFoundError, ValueError):
            return None
        if start is None:
            return None
        # Polar night: still dark at the end of the scan, so take 12 hours.
        return start, end if end is not None else start + timedelta(hours=12)

    def _sw_utc_day_dark_hours(
        self, place: spaceweather.Place, day: date, *, now: datetime
    ) -> tuple[datetime, datetime] | None:
        """The remaining dark hours that go with UTC ``day`` at ``place``.

        NOAA watch days are UTC days, and a local night straddles two of them.
        The local night (``day`` - 1 or ``day``) overlapping the UTC day most is
        the one meant; hours before ``now`` are dropped, and if that night is
        already over the next one is used. None when it never gets dark.
        """
        lo = datetime.combine(day, datetime.min.time(), UTC)
        hi = lo + timedelta(days=1)
        nights = [self._sw_dark_hours(place, day + timedelta(days=k)) for k in (-1, 0, 1)]
        best, best_overlap = None, timedelta(0)
        for i in (0, 1):
            window = nights[i]
            if window is None:
                continue
            overlap = min(window[1], hi) - max(window[0], lo)
            if overlap > best_overlap:
                best, best_overlap = i, overlap
        if best is None:
            return None
        for window in nights[best:]:
            if window is not None and window[1] > now:
                return max(window[0], now), window[1]
        return None

    def _sw_night_clouds(
        self, places: list[tuple[str, spaceweather.Place]], day: date, *, now: datetime
    ) -> dict[str, int | None]:
        """Mean cloud over each place's remaining dark hours for UTC ``day``."""
        clouds: dict[str, int | None] = {}
        for label, place in places:
            window = self._sw_utc_day_dark_hours(place, day, now=now)
            if window is None:
                clouds[label] = None  # no dark hours to report
                continue
            clouds[label] = self._sw_mean_cloud(place, *window, now=now)
        return clouds

    def _sw_poll_watches(
        self, channels: list[str], alerts: Any, *, now: datetime, deadline: float
    ) -> None:
        if not isinstance(alerts, list):
            raise spaceweather.InvalidPayload("alerts is not a list")
        state = self._sw_state
        # NOAA watch days are UTC days.
        levels = spaceweather.effective_levels(spaceweather.parse_watches(alerts), today=now.date())
        if not state.watches_seeded:
            state.announced_levels = {ch: dict(levels) for ch in channels}
            state.watches_seeded = True
            return
        told_by_channel = state.announced_levels
        for ch in list(told_by_channel):
            if ch not in channels:
                del told_by_channel[ch]
        min_level = self._SW_MIN_WATCH_LEVEL
        for ch in channels:
            if ch not in told_by_channel:
                told_by_channel[ch] = dict(levels)  # new channel: seed silently
                continue
            told = told_by_channel[ch]
            for d in [d for d in told if d < now.date()]:
                del told[d]
            news = spaceweather.watch_news(levels, told, min_level=min_level)
            if not news:
                continue
            if self._status_monotonic() > deadline or self._llm_executor.closing:
                return
            places, _failed = self._sw_channel_places(ch)
            clouds = self._sw_night_clouds(places, min(news), now=now)
            line = spaceweather.render_watch_line(news, clouds)
            if self._deliver_spaceweather_line(ch, line):
                told.update(news)

    def _sw_poll_episode(
        self,
        channels: list[str],
        samples: list[spaceweather.Sample],
        *,
        now: datetime,
        deadline: float,
    ) -> None:
        state = self._sw_state
        episode = spaceweather.advance_episode(
            state.episode,
            samples,
            threshold_nt=self._SW_BZ_THRESHOLD_NT,
            hours=self._SW_BZ_HOURS,
            now=now,
        )
        previous = state.episode
        if previous is not None and episode is None:
            state.last_episode, state.last_end = previous, now
        elif previous is None and episode is not None:
            # Resume or new storm is decided here, when an episode starts, and
            # never by elapsed time while none runs: a resumed storm is only
            # detected hours after Bz turns south again.
            last, last_end = state.last_episode, state.last_end
            if (
                last is not None
                and last_end is not None
                and episode.start - last_end <= self._SW_STORM_RESUME_WINDOW
            ):
                # A northward turn, not a new storm: keep the original start (so
                # its acks still hold and duration runs from it) and the deepest Bz.
                episode = replace(
                    episode, start=last.start, min_bz=min(episode.min_bz, last.min_bz)
                )
            else:
                state.episode_acked.clear()
                state.last_episode = state.last_end = None
        state.episode = episode
        seen, state.seen_channels = state.seen_channels, set(channels)
        if not state.episode_seeded:
            state.episode_acked = dict.fromkeys(channels, episode.start) if episode else {}
            state.episode_seeded = True
            return
        if episode is None:
            return  # acks kept: a resumed storm must stay told
        for ch in channels:
            if state.episode_acked.get(ch) == episode.start:
                continue
            if ch not in seen:  # arrived mid-episode: ack silently
                state.episode_acked[ch] = episode.start
                continue
            if self._status_monotonic() > deadline or self._llm_executor.closing:
                return
            places, failed = self._sw_channel_places(ch)
            if places:
                if not any(spaceweather.is_dark(p.lat, p.lon, now) for _l, p in places):
                    continue  # held until dark
            elif failed:
                continue  # lookup outage, nothing resolved: darkness unknown, retry
            # No usable places (none configured, or none match): always dark,
            # no cloud clause.
            direction = "south" if places and all(p.lat < 0 for _l, p in places) else "north"
            end = now + timedelta(hours=self._SW_STORM_CLOUD_HOURS)
            clouds = {label: self._sw_mean_cloud(p, now, end, now=now) for label, p in places}
            line = spaceweather.render_storm_line(
                episode, now=now, clouds=clouds, direction=direction
            )
            if self._deliver_spaceweather_line(ch, line):
                state.episode_acked[ch] = episode.start

    def _run_spaceweather_poll(self) -> None:
        """One announcer pass: storm watches, then storm conditions.

        Each feed fails alone: an unreadable alerts feed skips only the watch
        half, an unreadable solar wind only the episode half. The outer
        except is log control, as in _run_status_poll.
        """
        try:
            channels = self._sw_announce_channels()
            if not channels:
                return
            now = self._sw_now()
            deadline = self._status_monotonic() + self._SPACEWEATHER_PASS_BUDGET
            errors = (spaceweather.FetchError, spaceweather.InvalidPayload)
            try:
                alerts = self._sw_alerts(now)
                self._sw_poll_watches(channels, alerts, now=now, deadline=deadline)
            except errors as exc:
                self.log.info("spaceweather: alerts unavailable, skipping watches: %s", exc)
            try:
                samples = self._sw_samples(now)
            except errors as exc:
                self.log.info("spaceweather: solar wind unavailable, skipping storms: %s", exc)
            else:
                self._sw_poll_episode(channels, samples, now=now, deadline=deadline)
        except Exception as e:
            self.log.error("Space weather poll failed: %s", e)

    def _deliver_spaceweather_line(self, channel: str, line: str) -> bool:
        irc_conn = self._irc_for_channel(channel)
        if irc_conn is None:
            return False
        return self._safe_queue(irc_conn, self._safe_privmsg(channel, line))

    def _status_fetch_snapshot(
        self,
        source: str,
        *,
        timeout_cap: float | None = None,
        cached: statuspage.Snapshot | None = None,
    ) -> statuspage.Snapshot:
        """Fetch and strictly parse one status page.

        ``timeout_cap`` is the caller's remaining deadline budget. Without it a
        fetch entered near a deadline still runs its full ceiling, which is what
        made the first draft of the pass budget bound nothing.

        ``cached`` supplies the conditional-GET validators (ETag / Last-Modified).
        The poller passes nothing and gets its existing behaviour — validators
        from ``_status_read_cache``. A caller with its own cache (the query path)
        supplies its own snapshot so the refresh is still a conditional GET
        instead of an unconditional full fetch.

        Raises statuspage.FetchError or statuspage.InvalidPayload.
        """
        if cached is None:
            cached = self._status_read_cache.get(source)
        # min(...30): this borrows the LLM `timeout` registry key, which is
        # documented for LLM calls and may be raised by an operator for a slow
        # model. A poll must not hold an executor permit for that long — 30s is
        # the developer-tuned ceiling for one small status endpoint.
        timeout = min(self.registryValue("timeout"), 30)
        if timeout_cap is not None:
            timeout = min(timeout, max(1.0, timeout_cap))
        result = statuspage.fetch_summary(
            source,
            timeout=timeout,
            etag=cached.etag if cached else None,
            modified=cached.modified if cached else None,
            validate=validate_external_url,
            resolves_public=self.llm_service._resolves_to_public,
        )
        now = self._status_now()
        if result.not_modified:
            if cached is None:
                raise statuspage.FetchError("304 with no cached snapshot")
            return replace(cached, fetched_at=now)
        return statuspage.parse_summary(
            result.payload, fetched_at=now, etag=result.etag, modified=result.modified
        )

    def _status_fetch_now(
        self, source: str, *, deadline: float | None = None
    ) -> statuspage.Snapshot | None:
        """Refresh ONE source's read cache. Never touches lifecycle state.

        Called from the tool handler when that source's cache is cold or stale.
        Writing lifecycle state here would let a user's question consume an
        announcement: the poller would diff against a baseline that already
        contained the incident.

        The floor is per source — one page's recent read must not suppress
        another's. It stays an unlocked check-then-set: it is a cost guard, not
        a correctness guard, and a duplicate fetch is harmless.
        """
        now = self._status_now()
        if now - self._status_last_fetch.get(source, 0.0) < self._STATUS_FETCH_FLOOR:
            return self._status_read_cache.get(source)
        self._status_last_fetch[source] = now
        timeout_cap = None
        if deadline is not None:
            timeout_cap = deadline - self._status_monotonic()
            if timeout_cap <= self._STATUS_MIN_FETCH_WINDOW:
                return self._status_read_cache.get(source)
        try:
            snapshot = self._status_fetch_snapshot(source, timeout_cap=timeout_cap)
        except Exception as e:
            self.log.info("Status inline fetch failed for %s: %s", source, e)
            return self._status_read_cache.get(source)
        self._status_read_cache[source] = snapshot
        return snapshot

    def _status_query_snapshot(
        self, source: str, *, deadline: float | None = None
    ) -> statuspage.Snapshot | None:
        """Cached reading for a page we never poll.

        Writes only the two query dicts — never _status_state, never
        _status_read_cache. A queryable page has no lifecycle, so there is
        nothing to announce and nothing a question could consume.
        """
        now = self._status_now()
        cached = self._status_query_cache.get(source)
        if cached is not None and (now - cached.fetched_at) < self._STATUS_QUERY_TTL:
            return cached
        if now - self._status_query_failed_at.get(source, 0.0) < self._STATUS_HISTORY_RETRY:
            return cached
        timeout_cap = None
        if deadline is not None:
            timeout_cap = deadline - self._status_monotonic()
            if timeout_cap <= self._STATUS_MIN_FETCH_WINDOW:
                return cached
        try:
            snapshot = self._status_fetch_snapshot(source, timeout_cap=timeout_cap, cached=cached)
        except Exception as e:
            self.log.info("Status query fetch failed for %s: %s", source, e)
            self._status_query_failed_at[source] = now
            return cached
        self._status_query_cache[source] = snapshot
        self._status_query_failed_at.pop(source, None)
        self._status_evict_query_cache()
        return snapshot

    def _status_evict_query_cache(self) -> None:
        """Keep the newest _STATUS_QUERY_CACHE_MAX readings.

        Concurrent request threads can each land here after both computing
        the cache at (cap + 1) entries and independently pick the same
        oldest key; a plain ``del`` would raise KeyError for the loser and
        surface as "Could not read the service status page" to the model.
        ``pop(..., None)`` makes the second eviction of the same key a no-op.
        """
        excess = len(self._status_query_cache) - self._STATUS_QUERY_CACHE_MAX
        if excess <= 0:
            return
        oldest = sorted(self._status_query_cache.items(), key=lambda kv: kv[1].fetched_at)
        for source, _snap in oldest[:excess]:
            self._status_query_cache.pop(source, None)

    def _status_history_payload(self, source: str, *, deadline: float | None = None) -> list[dict]:
        """Lazily fetch and cache one source's resolved-incident history.

        Fetched ONLY when the model asks for history — never on the poll path,
        and never touching _status_state or _status_read_cache. Cached for
        _STATUS_HISTORY_TTL because resolved history changes rarely. Returns []
        on any failure or once the caller's deadline is spent; the caller
        reports current status regardless.

        A failed fetch is backed off for _STATUS_HISTORY_RETRY seconds: without
        this, every subsequent "when did it last go down" question during an
        outage retries a 30s fetch while holding an executor permit, even though
        the answer (still broken) hasn't changed.
        """
        now = self._status_now()
        cached = self._status_history_cache.get(source)
        if cached is not None and now - self._status_history_at.get(source, 0.0) < (
            self._STATUS_HISTORY_TTL
        ):
            return statuspage.to_history_payload(cached, now=now, limit=self._STATUS_HISTORY_LIMIT)
        if now - self._status_history_failed_at.get(source, 0.0) < self._STATUS_HISTORY_RETRY:
            if cached is not None:
                return statuspage.to_history_payload(
                    cached, now=now, limit=self._STATUS_HISTORY_LIMIT
                )
            return []
        timeout_cap = None
        if deadline is not None:
            timeout_cap = deadline - self._status_monotonic()
            if timeout_cap <= self._STATUS_MIN_FETCH_WINDOW:
                if cached is not None:
                    return statuspage.to_history_payload(
                        cached, now=now, limit=self._STATUS_HISTORY_LIMIT
                    )
                return []
        try:
            timeout = min(self.registryValue("timeout"), 30)
            if timeout_cap is not None:
                timeout = min(timeout, max(1.0, timeout_cap))
            result = statuspage.fetch_incidents(
                source,
                timeout=timeout,
                validate=validate_external_url,
                resolves_public=self.llm_service._resolves_to_public,
            )
            entries = statuspage.parse_incidents(result.payload)
        except Exception as e:
            self.log.info("Status history fetch failed for %s: %s", source, e)
            self._status_history_failed_at[source] = now
            if cached is not None:
                return statuspage.to_history_payload(
                    cached, now=now, limit=self._STATUS_HISTORY_LIMIT
                )
            return []
        self._status_history_cache[source] = entries
        self._status_history_at[source] = now
        self._status_history_failed_at[source] = 0.0
        return statuspage.to_history_payload(entries, now=now, limit=self._STATUS_HISTORY_LIMIT)

    def _status_source_entry(
        self, source: str, *, deadline: float, include_history: bool, polled: bool
    ) -> dict[str, Any]:
        """Build one source's tool-payload entry.

        ``polled`` selects the read path: a polled source refreshes through
        ``_status_read_cache`` under the staleness rule below, the same rule
        both before and after this was factored out of the aggregate loop. A
        queryable source is read through ``_status_query_snapshot``, which
        owns its own TTL and failure backoff and never touches lifecycle
        state.

        Every entry carries ``source``, the operator-configured host. That is
        the only identity available before a source's first successful fetch,
        and unlike ``service`` (the page's own name) it cannot be set by a
        third party — two pages both calling themselves "Claude" would otherwise
        be indistinguishable to the model.
        """
        entry: dict[str, Any] = {"source": self._status_host(source)}
        if polled:
            snapshot = self._status_read_cache.get(source)
            # `now` is captured per source, not once before the fan-out: a
            # source refreshed during another source's fetch would otherwise
            # have fetched_at > now, and snapshot_age_sec below would go
            # negative.
            now = self._status_now()
            if snapshot is None or (now - snapshot.fetched_at) > self._STATUS_STALE_AFTER:
                snapshot = self._status_fetch_now(source, deadline=deadline) or snapshot
                now = self._status_now()
        else:
            snapshot = self._status_query_snapshot(source, deadline=deadline)
            now = self._status_now()
        if snapshot is None:
            # The poller-facing wording ("has not been read yet") implies a
            # scheduled read is pending; a queryable page is never polled, so
            # there is nothing to wait for — only a failed on-demand fetch.
            entry["error"] = (
                "This status page has not been read yet."
                if polled
                else "This status page could not be reached."
            )
            return entry
        entry["service"] = (
            statuspage.sanitise_text(statuspage.strip_urls(snapshot.page_name), limit=60)
            or entry["source"]
        )
        # The per-snapshot note is dropped and stated once at the top level:
        # repeating it per service is pure token cost.
        entry.update(
            {k: v for k, v in statuspage.to_tool_payload(snapshot, now=now).items() if k != "note"}
        )
        if (now - snapshot.fetched_at) > self._STATUS_STALE_AFTER:
            entry["stale"] = True
            entry["error"] = "This status page is currently unreachable; this is the last reading."
        if include_history:
            entry["recent_incidents"] = self._status_history_payload(source, deadline=deadline)
        return entry

    def _status_single_payload(
        self, source: str, *, include_history: bool, polled: bool
    ) -> dict[str, Any]:
        """Build the payload for exactly one named source (polled or queryable)."""
        deadline = self._status_monotonic() + self._STATUS_TOOL_BUDGET
        entry = self._status_source_entry(
            source, deadline=deadline, include_history=include_history, polled=polled
        )
        payload: dict[str, Any] = {"services": [entry], "note": statuspage.UNTRUSTED_NOTE}
        if "error" in entry:
            payload["error"] = entry["error"]
        return payload

    def _status_tool_payload(
        self,
        *,
        service: str | None = None,
        include_history: bool = False,
        pages: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Build the model-facing payload.

        ``service`` omitted: one entry per configured POLLED source (today's
        behaviour, pinned by production — "are Claude and Codex up?" answers
        in one call). ``pages`` is not consulted on this path — the polled
        list always comes from ``_status_sources()`` — so it constrains
        nothing here; pass it only alongside ``service``.

        ``service`` named: one entry for that page alone, polled or
        queryable, resolved from ``pages`` when the caller supplies the
        frozen name->source mapping the tool schema was built from —
        resolving live here instead would let config churn between the
        model's call and its dispatch route the call somewhere the enum
        never advertised. An unresolvable name still returns the polled set,
        plus a top-level ``error`` naming it: service.py records any dict
        without an "error" key as a successful tool call, so a bare note
        would let the model answer about the wrong service.

        Reads (and may refresh) the read/query caches only. Lifecycle state
        is the poller's alone. Current status is always returned regardless
        of ``include_history`` — history is additive.
        """
        if service is not None:
            named = pages if pages is not None else self._status_named_pages(warn=False)
            source = next(
                (s for n, s in named.items() if n.lower() == service.strip().lower()), None
            )
            if source is None:
                # `pages` is not forwarded here: the service is None branch
                # below always resolves the polled list from
                # _status_sources(), so a frozen mapping has nothing to
                # constrain on this path.
                payload = self._status_tool_payload(include_history=include_history)
                unresolved = (
                    f"No status page named {statuspage.sanitise_text(service, limit=40)!r} "
                    "is configured."
                )
                existing = payload.get("error")
                if payload.get("services") and existing is None:
                    # The aggregate read succeeded (at least in part) and has
                    # nothing to say for itself yet: describe what the
                    # `services` list actually holds.
                    payload["error"] = f"{unresolved} The services listed are the ones that are."
                elif existing:
                    # The aggregate already has its own diagnostic — e.g. every
                    # polled page was unreadable — which the "services listed
                    # are the ones that are" wording would falsely contradict.
                    # Append rather than overwrite so both are said.
                    payload["error"] = f"{unresolved} {existing}"
                else:
                    # No polled sources either: there is no list to point at,
                    # so there is nothing to append to.
                    payload["error"] = unresolved
                return payload
            return self._status_single_payload(
                source,
                include_history=include_history,
                polled=source in set(self._status_sources(warn=False)),
            )

        # warn=False: this runs per tool call, not per poll; the poller
        # already logs a bad entry on its own ~2-minute cadence.
        sources = self._status_sources(warn=False)
        if not sources:
            return {"error": "No status pages are configured."}

        deadline = self._status_monotonic() + self._STATUS_TOOL_BUDGET
        services: list[dict[str, Any]] = []
        readable = 0
        for source in sources:
            entry = self._status_source_entry(
                source, deadline=deadline, include_history=include_history, polled=True
            )
            if "error" not in entry:
                readable += 1
            services.append(entry)

        payload: dict[str, Any] = {"services": services, "note": statuspage.UNTRUSTED_NOTE}
        if readable == 0:
            # service.py:5557 treats a top-level dict with no "error" key as a
            # successful tool call, so an all-failed list must say so out loud.
            payload["error"] = "No configured status page could be read."
        return payload

    def _status_rotate(self, sources: list[str]) -> list[str]:
        """Order sources for this pass, resuming where the last one stopped.

        A cursor whose source is no longer configured falls back to the head —
        this is why the cursor is a canonical id and not an index: an index
        silently points at a different page after any reorder or removal.
        """
        cursor = self._status_cursor
        if cursor is None or cursor not in sources:
            return list(sources)
        i = sources.index(cursor)
        return list(sources[i:]) + list(sources[:i])

    def _poll_one_source(self, source: str, *, deadline: float, lines_left: int) -> int:
        """Fetch, classify and announce one source. Returns lines delivered."""
        self._status_last_fetch[source] = self._status_now()
        snapshot = self._status_fetch_snapshot(
            source, timeout_cap=deadline - self._status_monotonic()
        )
        self._status_read_cache[source] = snapshot
        delta, new_state = statuspage.classify(
            self._status_state.get(source, statuspage.StatusState()),
            snapshot,
            max_opened=self._STATUS_MAX_ANNOUNCE_PER_POLL,
        )
        self._status_state[source] = new_state
        if delta.discarded:
            self.log.warning(
                "Status poll discarded %i opened incidents past the per-poll cap for %s",
                delta.discarded,
                source,
            )
        # Both branches, not just openings: _announce_status walks delta.resolved
        # too, and gating on delta.opened alone meant an incident that cleared in
        # a pass where nothing new opened sat in pending_resolved unspoken — then
        # surfaced as a stale all-clear alongside the next unrelated opening.
        if not (delta.opened or delta.resolved) or lines_left <= 0:
            return 0
        if self._llm_executor.closing:
            return 0
        return self._announce_status(
            source,
            delta,
            snapshot,
            lines_left=lines_left,
            template_only=(deadline - self._status_monotonic()) < self._STATUS_REWRITE_RESERVE,
        )

    def _run_status_poll(self) -> None:
        """Poll every configured source under one wall-clock budget.

        Sequential inside the worker's single permit: no submit (raises
        RecursiveSubmitError from worker context) and no nested permit (double
        acquire). The try/except is for log control only — schedule.py already
        catches and re-arms (schedule.py:118-122, :150-153).
        """
        try:
            sources = self._status_sources()
            queryable = [s for s in self._status_named_pages().values() if s not in set(sources)]
            self._status_prune_sources(sources, queryable)
            if not sources:
                return
            deadline = self._status_monotonic() + self._STATUS_PASS_BUDGET
            lines_left = self._STATUS_MAX_LINES_PER_POLL
            rotated = self._status_rotate(sources)
            for idx, source in enumerate(rotated):
                if self._llm_executor.closing:
                    return
                if deadline - self._status_monotonic() <= self._STATUS_MIN_FETCH_WINDOW:
                    # Never started, so the cursor must NOT advance past it:
                    # this is the source the next pass owes a poll.
                    self._status_cursor = source
                    return
                try:
                    lines_left -= self._poll_one_source(
                        source, deadline=deadline, lines_left=lines_left
                    )
                except statuspage.FetchError as e:
                    # Transient and self-healing: network blip, timeout, a
                    # 5xx. The next poll retries on its own.
                    self.log.info(
                        "Status poll failed for %s, retaining last good state: %s", source, e
                    )
                except statuspage.InvalidPayload as e:
                    # Structural, not transient: the page's vocabulary moved
                    # under a still-strict guard (bad indicator, non-string
                    # status, etc). A page that parsed yesterday and rejects
                    # today needs a human, not a retry.
                    self.log.warning(
                        "Status poll rejected payload for %s, retaining last good state: %s",
                        source,
                        e,
                    )
                except Exception as e:
                    self.log.error("Status poll raised for %s: %s", source, e)
                finally:
                    # Advance past an ATTEMPTED source even when it raised, so a
                    # permanently broken page cannot pin the head of the rotation
                    # and starve everything behind it.
                    self._status_cursor = rotated[idx + 1] if idx + 1 < len(rotated) else None
        except Exception as e:
            self.log.error("Status poll pass raised: %s", e)

    def _status_announce_budget_ok(self) -> bool:
        """Token bucket for announcement completions.

        Every other unattended fire in this repo is metered
        (_unattended_ask_rate_limited); the announcer has no user and so
        inherits no bucket. Over budget falls through to the template, which
        costs no completion — the channel still hears about the outage.
        """
        now = self._status_now()
        self._status_announce_times = [t for t in self._status_announce_times if now - t < 3600]
        return len(self._status_announce_times) < self._STATUS_ANNOUNCE_MAX_PER_HOUR

    def _status_rewrite_ok(self, text: str, *, allowed_host: str, label: str) -> bool:
        """Post-check an LLM rewrite before it reaches an unprompted channel line.

        tools=[] stops tool calls; it does not stop the bot repeating a
        phishing link in its own voice with nobody having asked. The URL host
        check is the highest-value filter on this path.

        ``allowed_host`` is caller-derived from operator config (the
        canonical source, via ``_status_host``) only, never from the fetched
        payload — a hostile status page could otherwise set its own
        ``page_url`` and become the only host this gate permits. ``label``
        MAY be quoted from the fetched payload (``page_name``), but only
        after the caller has stripped URLs and sanitised it (see
        ``_announce_status``) — this check only
        confirms the label appears in the text, so it never on its own
        authorizes a URL. An empty ``label`` fails closed (rejects) rather
        than skipping the service-name check. Never raises: a malformed URL
        in the rewrite is treated as a rejection, not an exception that would
        abort the rest of the announce pass.
        """
        if not text or not text.strip():
            return False
        if not label or label.lower() not in text.lower():
            return False
        for match in statuspage.URL_LIKE_RE.finditer(text):
            token = match.group(0)
            candidate = token if "://" in token else f"//{token}"
            try:
                host = urlparse(candidate).hostname or ""
            except ValueError:
                return False
            if host.lower() != allowed_host.lower():
                return False
        return True

    def _status_rewrite(
        self,
        incident: statuspage.IncidentView,
        channel: str,
        *,
        snapshot: statuspage.Snapshot,
        url: str,
        label: str,
        event: str = "opened",
        duration_sec: int | None = None,
    ) -> str | None:
        """One-shot rewrite of the sanitised incident facts in channel voice.

        Runs INLINE in the poll worker's existing permit — no submit (raises
        RecursiveSubmitError from worker context, executor.py:102-106) and no
        nested permit (double acquire; self-deadlock at
        maxConcurrentLLMCalls=1).

        ``url`` is built from the operator-configured value (``statusPageUrl``
        plus the incident's own path segment via ``statuspage.incident_url``),
        never re-read from the fetched snapshot: the page's own ``page_url``
        is third-party data, and handing the model a hostile page's own URL
        would make ``_status_rewrite_ok`` reject every rewrite against the
        wrong host. ``label`` is derived from the third-party ``page_name``
        (sanitised and URL-stripped, falling back to the configured host)
        — it is quoted from the payload, not caller-derived.
        """
        if snapshot is None:
            return None
        resolved = event == "resolved"
        facts = {
            "event": event,
            "name": statuspage.sanitise_text(incident.name),
            # An incident that vanished from the unresolved set carries its
            # last LIVE status ("investigating") and its last live update
            # body. Handing either to a resolution rewrite invites a line
            # that announces an outage as ongoing and over in one breath, so
            # the status is stated as the event and the stale body dropped.
            "status": "resolved" if resolved else incident.status,
            "impact": statuspage.sanitise_text(incident.impact),
            "affected_components": [
                statuspage.sanitise_text(c) for c in incident.affected_components
            ],
            "service": label,
            "url": url,
        }
        if resolved:
            # Formatted, not raw seconds. Handed the integer the model narrates
            # it verbatim — "resolved after lasting 2,170 seconds" reached
            # #clanker on 2026-08-15 while the template it was upgrading said
            # "36m" for the same incident. Omitted entirely when unusable, so
            # the model has nothing to narrate rather than an empty string to
            # invent around; that matches what the template does.
            human = statuspage.format_duration(duration_sec)
            if human:
                facts["duration"] = human
        else:
            facts["latest_update"] = statuspage.sanitise_text(incident.latest_update_body)
        try:
            return self.llm_service.status_announce_completion(facts=facts, channel=channel)
        except Exception as e:
            self.log.info("Status rewrite failed, using template: %s", e)
            return None

    def _irc_for_channel(self, channel: str):
        """Return the Irc whose state currently holds *channel*, else None."""
        for irc_conn in list(world.ircs):
            # IrcDict's own lookup folds case; list() would yield lowered keys.
            if channel in irc_conn.state.channels:
                return irc_conn
        return None

    def _deliver_status_line(
        self,
        incident: statuspage.IncidentView,
        *,
        template: str,
        by_overlay: dict[str, list[str]],
        snapshot: statuspage.Snapshot,
        label: str,
        configured_host: str,
        link: str,
        event: str,
        duration_sec: int | None = None,
        template_only: bool = False,
    ) -> bool:
        """Send one incident line to every deliverable channel.

        Returns True if at least one channel took it, which is the caller's
        cue to mark the event announced. Shared by the opened and resolved
        branches so an all-clear inherits the same budget, post-checks and
        truncation as the opening it closes.
        """
        delivered = False
        for group_channels in by_overlay.values():
            # Look up deliverability BEFORE spending a completion or the
            # hourly budget: a channel the bot has since parted cannot be
            # delivered to regardless of what the rewrite produces.
            deliverable = [
                (channel, irc_conn)
                for channel in group_channels
                if (irc_conn := self._irc_for_channel(channel)) is not None
            ]
            if not deliverable:
                continue

            text = template
            if not template_only and self._status_announce_budget_ok():
                # First deliverable channel drives provider cache routing;
                # the rewrite itself is shared across the whole group.
                rewrite = self._status_rewrite(
                    incident,
                    deliverable[0][0],
                    snapshot=snapshot,
                    url=link,
                    label=label,
                    event=event,
                    duration_sec=duration_sec,
                )
                self._status_announce_times.append(self._status_now())
                if rewrite and self._status_rewrite_ok(
                    rewrite, allowed_host=configured_host, label=label
                ):
                    text = rewrite

            safe = self.llm_service.sanitize_output(text)
            safe = self._collapse_for_irc(safe) or safe
            if len(safe) > self._STATUS_ANNOUNCE_MAX_LEN:
                safe = safe[: self._STATUS_ANNOUNCE_MAX_LEN].rsplit(" ", 1)[0]
            if not safe:
                # render_line fails closed to "" on pathological input, and the
                # whitespace truncation above can empty a single-token line. Skip
                # rather than queue "PRIVMSG #chan :" — and leave the incident
                # unmarked so the next poll retries it.
                continue

            for channel, irc_conn in deliverable:
                if self._safe_queue(irc_conn, self._safe_privmsg(channel, safe)):
                    delivered = True
        return delivered

    def _announce_status(
        self,
        source: str,
        delta: statuspage.Delta,
        snapshot: statuspage.Snapshot,
        *,
        lines_left: int,
        template_only: bool = False,
    ) -> int:
        """Announce one source's openings and all-clears. Returns lines sent.

        Template-primary: the deterministic line is built first and is always
        available. The rewrite is an upgrade, applied only when the budget
        allows, the pass has time, and every post-check passes.

        An incident is marked announced only after a successful queue, so a
        drop during shutdown — or an incident pushed past ``lines_left`` — is
        retried on the next poll. Openings and all-clears are tracked in
        separate maps, so one incident produces at most one of each over the
        process lifetime.

        ``snapshot`` is the observation the caller classified, passed in
        rather than re-read from ``_status_read_cache``: that cache is
        writable by the tool path, so re-reading it can label a delta with a
        different reading.
        """
        if lines_left <= 0:
            return 0

        # Copy before iterating: stock RSS copies (RSS/plugin.py:405) because
        # channel state mutates under JOIN/PART on the IRC thread, and outages
        # are exactly when churn peaks.
        channels = [
            channel
            for channel in sorted(self._all_known_channels())
            if self.registryValue("statusAnnounce", channel)
        ]
        if not channels:
            return 0

        # allowed_host/label are derived from OPERATOR CONFIG (the canonical
        # source), never from the fetched payload: page_name/page_url in the
        # snapshot are third-party data, and trusting them here would let a
        # hostile status page nominate its own phishing host as the only one
        # this gate permits. label itself IS quoted from the payload
        # (page_name), so it is URL-stripped and sanitised before use.
        configured_host = self._status_host(source)
        label = (
            statuspage.sanitise_text(statuspage.strip_urls(snapshot.page_name), limit=60)
            or configured_host
            or "Status"
        )

        # The rewrite varies only with the channel's assistantSystemPrompt
        # overlay, so channels sharing one share a completion. This both cuts
        # cost and removes the deterministic starvation of alphabetically-later
        # channels once the hourly budget is exhausted.
        by_overlay: dict[str, list[str]] = {}
        for channel in channels:
            overlay = self.registryValue("assistantSystemPrompt", channel) or ""
            by_overlay.setdefault(overlay, []).append(channel)

        sent = 0
        for incident in delta.opened:
            if sent >= lines_left:
                break
            if self._deliver_status_line(
                incident,
                template=statuspage.render_line(incident, page_name=label, page_url=source),
                by_overlay=by_overlay,
                snapshot=snapshot,
                label=label,
                configured_host=configured_host,
                link=statuspage.incident_url(source, incident.id),
                event="opened",
                template_only=template_only,
            ):
                sent += 1
                self._status_state[source] = statuspage.mark_announced(
                    self._status_state.get(source, statuspage.StatusState()),
                    incident.id,
                    now=self._status_now(),
                )

        for incident in delta.resolved:
            if sent >= lines_left:
                break
            duration = statuspage.incident_duration_sec(incident, now=self._status_now())
            if self._deliver_status_line(
                incident,
                template=statuspage.render_resolved_line(
                    incident, page_name=label, page_url=source, duration_sec=duration
                ),
                by_overlay=by_overlay,
                snapshot=snapshot,
                label=label,
                configured_host=configured_host,
                link=statuspage.incident_url(source, incident.id),
                event="resolved",
                duration_sec=duration,
                template_only=template_only,
            ):
                sent += 1
                self._status_state[source] = statuspage.mark_resolved_announced(
                    self._status_state.get(source, statuspage.StatusState()),
                    incident.id,
                    now=self._status_now(),
                )

        return sent

    # Safety poll interval (seconds) — fallback for event-driven wakeups
    _SAFETY_POLL_INTERVAL = 300  # 5 minutes

    # Status page polling. Constants rather than registry keys, matching
    # _SAFETY_POLL_INTERVAL: one small endpoint, tuned by the developer.
    _STATUS_POLL_INTERVAL = 120
    _STATUS_MAX_ANNOUNCE_PER_POLL = 3
    _STATUS_ANNOUNCE_MAX_PER_HOUR = 6
    _STATUS_FETCH_FLOOR = 30
    _STATUS_MIN_FETCH_WINDOW = 2.0
    _STATUS_ANNOUNCE_MAX_LEN = 400
    _STATUS_HISTORY_TTL = 3600  # history changes rarely; 1 hour is plenty
    _STATUS_HISTORY_LIMIT = 5
    _STATUS_HISTORY_RETRY = 120  # backoff before retrying a failed history fetch
    _STATUS_MAX_SOURCES = 5
    _STATUS_MAX_QUERYABLE = 20
    # Queryable pages are refreshed only when asked for, so their TTL is
    # shorter than the 600s staleness line — nothing else refreshes them.
    _STATUS_QUERY_TTL = 300
    # Equal to _STATUS_MAX_QUERYABLE on purpose, expressed as a reference so
    # the coupling can't drift: a cache smaller than the allowlist thrashes —
    # cycling every page inside the TTL evicts each entry before it is
    # reused, so every request fetches despite the cache.
    _STATUS_QUERY_CACHE_MAX = _STATUS_MAX_QUERYABLE

    # Whole-call budget for the tool path, covering current-status refreshes and
    # the history fan-out together. The tool runs inside the asking request's
    # permit, so an unbounded fan-out makes that user wait minutes.
    _STATUS_TOOL_BUDGET = 20
    # A source is reported stale against its own last SUCCESSFUL read. Fixed
    # rather than derived from the poll interval: with rotation and a pass
    # budget, a healthy source can legitimately wait several passes, and
    # 2 * interval would label it unreachable.
    _STATUS_STALE_AFTER = 600

    # Whole-pass wall-clock budget. A pass walks every configured source inside
    # one executor permit, so without this N sources multiply the permit hold by
    # N. The deadline is propagated into each fetch's timeout and into the
    # decision to skip rewrites — checking it only between sources bounds
    # nothing, since a fetch entered at t=44 still runs its full ceiling.
    _STATUS_PASS_BUDGET = 45

    # Storm announcer: one pass every 5 minutes; a pass stops sending once it
    # has run this many seconds (geocodes and cloud fetches are sequential).
    _SPACEWEATHER_POLL_INTERVAL = 300
    _SPACEWEATHER_PASS_BUDGET = 60
    _SW_STORM_CLOUD_HOURS = 3
    # An episode starting this soon after the last one ended is the same storm.
    _SW_STORM_RESUME_WINDOW = timedelta(hours=3)
    # Below this much remaining, the pass stops spending completions and posts
    # templates. The template has always been the primary path and the rewrite
    # an upgrade, so nothing is lost but prose.
    _STATUS_REWRITE_RESERVE = 20
    # Global burst cap across all sources for one pass. Per-source caps are 3
    # openings (max_opened) plus 3 all-clears (classify's max_resolved default),
    # so five sources could otherwise emit 30 unprompted lines at once.
    _STATUS_MAX_LINES_PER_POLL = 5

    # Delivery retry constants: 15 * 2^attempt, capped at 120s, max 10 attempts
    _DELIVERY_BASE_BACKOFF = 15
    _DELIVERY_MAX_BACKOFF = 120
    _DELIVERY_MAX_ATTEMPTS = 10

    def _deliver_pending_result(self, r) -> None:
        """Deliver a single pending task result to the correct target.

        Sends the message to the original channel or PM nick.  For results
        with a ``task_id`` (from the durable delivery queue), acknowledges
        successful delivery by deleting the row, or retries with bounded
        exponential backoff on failure.

        Args:
            r: PendingTaskResult from check_pending_tasks.
        """
        target = r.reply_target
        nick = r.nick
        prompt_preview = self.llm_service.sanitize_output(r.prompt_preview)

        if r.status == "expired":
            text = f'{nick}: sorry, your {r.task_type} request "{prompt_preview}" expired.'
            self.log.warning(
                "Deferred task expired: task_type=%s nick=%s target=%s prompt=%s",
                r.task_type,
                nick,
                target,
                prompt_preview[:50],
            )
        elif r.status == "failed_terminal":
            reason = self.llm_service.sanitize_output(r.reason)[:200]
            text = f'{nick}: sorry, your {r.task_type} request "{prompt_preview}" failed: {reason}'
            self.log.warning(
                "Deferred task failed_terminal: task_type=%s nick=%s target=%s reason=%s",
                r.task_type,
                nick,
                target,
                reason[:100],
            )
        elif r.status == "completed":
            content = self.llm_service.sanitize_output(r.content)
            if r.task_type == "draw":
                text = f'{nick}: your image is ready! "{prompt_preview}" \u2192 {content}'
            else:
                # ask or fallback (incl. recovered verse, which is unbounded —
                # verse timeouts recover under the "ask" task_type). Long content
                # is saved to the HTTP server and replaced with a teaser + URL,
                # mirroring the live _send_long_reply path; without this a
                # multi-paragraph scene becomes one oversized PRIVMSG the server
                # silently truncates. Short content stays inline (collapsed below).
                text = self._format_pending_completed_reply(nick, content, target)
        else:
            return

        # Pending-task delivery bypasses _send_long_reply, so collapse multi-line
        # content. A raw \n on PRIVMSG triggers Excess Flood disconnects.
        text = self._collapse_for_irc(text) or text

        # Try to deliver via IRC. _safe_queue serializes workers and short-
        # circuits cleanly when shutting down — a False return must not
        # advance durable delivery state.
        delivered = False
        try:
            for irc_conn in world.ircs:
                # Thread the answer under the original request when the server
                # negotiated message-tags and we kept the msgid. Degrades to a
                # plain PRIVMSG otherwise — the tag is an improvement to how
                # the line is displayed, never a precondition for sending it.
                reply_to = (
                    r.reply_msgid
                    if r.reply_msgid and irc_has_caps(irc_conn, "message-tags")
                    else None
                )
                if r.is_channel:
                    if target in irc_conn.state.channels:
                        if self._safe_queue(irc_conn, self._safe_privmsg(target, text, reply_to)):
                            delivered = True
                        break
                else:
                    # PM delivery — use first available connection
                    if self._safe_queue(irc_conn, self._safe_privmsg(target, text, reply_to)):
                        delivered = True
                    break
        except Exception as e:
            self.log.warning(
                "queueMsg failed for task_id=%s: %s",
                r.task_id,
                self.llm_service.sanitize_output(str(e)),
            )
            delivered = False

        # Acknowledge or retry delivery for durable results.
        #
        # A successful send MUST be acked even if shutdown began after the
        # message went out — otherwise the row survives and is re-delivered
        # next process lifetime (duplicate IRC send). Only the not-delivered
        # branch (retry-state writes + wakeup) and usage logging are skipped
        # while closing, since nothing was sent and leaving the row for the
        # next lifetime is the correct, mutation-free behavior.
        if r.task_id is not None:
            if delivered:
                # Booked before the ack: if the delete below fails and the row
                # re-delivers, the task's usage_booked flag stops a second row.
                # A delivered failure still billed its retry (an empty answer).
                self._book_pending_usage(
                    r.task_id,
                    status="success" if r.status == "completed" else "error",
                    error_detail="" if r.status == "completed" else r.reason,
                )
                # Best-effort ack: a transient delete failure (e.g. DB lock)
                # after a successful send must not bubble up as a misleading
                # "delivery failed". The row simply re-delivers next tick
                # (at-least-once), which beats losing a delivered result.
                try:
                    self.db.delete_pending_task(r.task_id)
                except Exception as e:
                    self.log.warning(
                        "Delivered task_id=%s but ack(delete) failed; may re-deliver: %s",
                        r.task_id,
                        self.llm_service.sanitize_output(str(e)),
                    )
            elif not self._llm_executor.closing:
                now = time.time()
                attempt = max(r.delivery_attempt_count, 0) + 1
                delay = exponential_backoff(
                    attempt - 1, base=self._DELIVERY_BASE_BACKOFF, cap=self._DELIVERY_MAX_BACKOFF
                )
                state = "delivery_failed" if attempt >= self._DELIVERY_MAX_ATTEMPTS else "retrying"
                retry_at = now + delay
                self.db.update_delivery_attempt(
                    task_id=r.task_id,
                    delivery_state=state,
                    last_delivery_error="IRC delivery failed",
                    delivery_attempt_count=attempt,
                    next_attempt_at=retry_at,
                )
                if state == "delivery_failed":
                    self._book_pending_usage(
                        r.task_id, status="error", error_detail="delivery failed"
                    )
                else:
                    self._schedule_queue_wakeup(at_time=retry_at)

    def _remember_own_msgid(self, irc: callbacks.Irc, msg: IrcMsg) -> None:
        """Record the msgid of an echoed bot line (PRIVMSG, NOTICE, or the
        BATCH opener of a multiline reply, which is what clients reply to).

        Needs echo-message; without it nothing is recorded and replies to the
        bot stay unaddressed, as before.
        """
        if msg.command not in ("PRIVMSG", "NOTICE", "BATCH") or not msg.prefix:
            return
        if not ircutils.isUserHostmask(msg.prefix) or not ircutils.strEqual(irc.nick, msg.nick):
            return
        msgid = (getattr(msg, "server_tags", None) or {}).get("msgid")
        if not msgid:
            return
        with self._own_msgids_lock:
            self._own_msgids[(irc.network, msgid)] = None
            while len(self._own_msgids) > _OWN_MSGID_CAP:
                self._own_msgids.popitem(last=False)
        # delete_last_reply's view: the bot's recent lines per target. A
        # multiline reply is one BATCH (its opener's msgid redacts the lot).
        if msg.command == "BATCH":
            target = msg.args[2] if len(msg.args) >= 3 else ""
        else:
            target = msg.args[0] if msg.args else ""
        if not target:
            return
        own_lines = getattr(self, "_own_lines", None)
        if own_lines is None:
            own_lines = self._own_lines = {}
        key = (irc.network, ircutils.toLower(target))
        with self._own_msgids_lock:
            lines = own_lines.setdefault(key, collections.deque(maxlen=_OWN_LINES_PER_TARGET))
            lines.append((msgid, time.time()))

    def _remember_last_msgid(self, irc: callbacks.Irc, msg: IrcMsg) -> None:
        """Record the msgid of each channel PRIVMSG under its sender, so the
        react tool can point at "bob's last message". The bot's own echoed
        lines count too: reacting to them is a fair ask.
        """
        if msg.command != "PRIVMSG" or len(msg.args) < 2 or not msg.prefix:
            return
        channel = msg.args[0]
        if not ircutils.isChannel(channel) or not ircutils.isUserHostmask(msg.prefix):
            return
        msgid = (getattr(msg, "server_tags", None) or {}).get("msgid")
        if not msgid:
            return
        key = (irc.network, ircutils.toLower(channel), ircutils.toLower(msg.nick))
        with self._last_msgids_lock:
            self._last_msgids[key] = msgid
            self._last_msgids.move_to_end(key)
            while len(self._last_msgids) > _LAST_MSGID_CAP:
                self._last_msgids.popitem(last=False)
            self._last_msgids_dirty[key] = (msgid, time.time())

    def _load_last_msgids(self) -> None:
        """Refill the react tool's msgid LRU from the database, so a restart
        does not forget everyone who spoke before it."""
        try:
            rows = self.db.load_last_msgids(_LAST_MSGID_CAP)
        except Exception:
            self.log.exception("last_msgids load failed; starting empty")
            return
        with self._last_msgids_lock:
            for network, channel, nick, msgid in rows:
                self._last_msgids[(network, channel, nick)] = msgid

    def _flush_last_msgids(self) -> None:
        """Write LRU entries changed since the last flush. Runs every minute
        and from die(), so a crash loses at most a minute of lines."""
        with self._last_msgids_lock:
            dirty, self._last_msgids_dirty = self._last_msgids_dirty, {}
        if not dirty:
            return
        rows = [(n, c, k, m, ts) for (n, c, k), (m, ts) in dirty.items()]
        try:
            self.db.save_last_msgids(rows, keep=_LAST_MSGID_CAP)
        except Exception:
            self.log.exception("last_msgids flush failed (%s rows)", len(rows))

    def _is_reply_to_own_line(self, irc: callbacks.Irc, msg: IrcMsg) -> bool:
        """True when msg carries +draft/reply pointing at one of the bot's lines."""
        target = (getattr(msg, "server_tags", None) or {}).get("+draft/reply")
        if not target:
            return False
        with self._own_msgids_lock:
            return (irc.network, target) in self._own_msgids

    def inFilter(self, irc: callbacks.Irc, msg: IrcMsg) -> IrcMsg | None:  # noqa: N802
        """Sanitize PRIVMSG text before Limnoria's tokenizer processes it.

        Limnoria's command tokenizer interprets ``[…]`` as nested-command
        syntax and raises ``SyntaxError`` on unmatched brackets.  Messages
        containing ANSI escape sequences (e.g. ``\\x1b[6n``) or casual
        bracket use (e.g. ``array[0``) crash the tokenizer before
        ``invalidCommand`` ever runs.

        This filter:
        1. Strips C0 control characters (except TAB/LF/CR) — removes the
           ESC byte from ANSI sequences.
        2. Replaces ``[`` and ``]`` with full-width equivalents when brackets
           are unbalanced — prevents the tokenizer crash while keeping the
           text readable for the LLM.
        """
        if self._capture_history(irc, msg):
            return None
        self._remember_own_msgid(irc, msg)
        self._remember_last_msgid(irc, msg)
        if msg.command == "NOTICE":
            self._capture_ctcp_time(irc, msg)
            self._capture_ctcp_reply(irc, msg)
        if msg.command != "PRIVMSG" or len(msg.args) < 2:
            return msg

        text = msg.args[1]
        # A CTCP ACTION is scrubbed inside its frame: the \x01 delimiters sit
        # in _CTRL_CHAR_RE's range, and stripping them would turn "/me waves"
        # into a plain message reading "ACTION waves".
        action_body = self._action_payload(msg)
        raw = text if action_body is None else action_body
        cleaned = _CTRL_CHAR_RE.sub("", raw)

        # Escape unbalanced brackets that would crash the tokenizer
        if cleaned.count("[") != cleaned.count("]"):
            cleaned = cleaned.replace("[", "\uff3b").replace("]", "\uff3d")

        if cleaned != raw:
            rewritten = cleaned if action_body is None else f"\x01ACTION {cleaned}\x01"
            msg = ircmsgs.IrcMsg(msg=msg, args=(msg.args[0], rewritten))
            text = rewritten

        # Gate Limnoria's command dispatcher: only messages prefixed with
        # the configured command character reach Owner.doPrivmsg's
        # tokenizer. Nick-addressed channel messages and unprefixed PMs are
        # routed through the assistant in doPrivmsg instead — otherwise
        # plain English verbs like "remove", "later", or "search" collide
        # with built-in plugins and surface ambiguity errors. We tag
        # msg.addressed=''; callbacks.addressed() reads the cached tag,
        # returns '', and Owner.doPrivmsg skips dispatch.
        if msg.prefix and not ircutils.strEqual(irc.nick, msg.nick):
            prefix_chars = conf.supybot.reply.whenAddressedBy.chars()
            if not text or text[0] not in prefix_chars:
                msg.tag("addressed", "")

        return msg

    def doPrivmsg(self, irc: callbacks.Irc, msg: IrcMsg) -> None:  # noqa: N802
        """Route addressed text to the assistant; observe channel chatter for context.

        Two paths:

        - Addressed text (channel message starting with our nick, or any PM
          without the configured command-prefix character) goes to
          ``_ask_impl``. The Limnoria command dispatcher is suppressed for
          these by ``inFilter``.
        - Other channel messages flow through the existing
          ``privacyOptInOnly`` capture logic.

        Explicit prefix-char commands (e.g. ``@search`` or ``@later``) are
        handled entirely by Limnoria's dispatcher and short-circuit before
        the assistant routing.
        """
        # Universal early-outs before any branching.
        if self._is_old_message(msg):
            return
        if ircmsgs.isCtcp(msg) and not ircmsgs.isAction(msg):
            return
        if not msg.prefix or ircutils.strEqual(irc.nick, msg.nick):
            return
        # Server-originated PRIVMSGs (e.g. some services responses) carry
        # a server prefix instead of nick!user@host. Downstream code
        # (preflight, account resolution, rate-limit identity) calls
        # ``nickFromHostmask`` which asserts user-hostmask form, so drop
        # these here rather than letting every call site guard.
        if not ircutils.isUserHostmask(msg.prefix):
            return

        text = msg.args[1] if len(msg.args) > 1 else ""
        if not text:
            return

        # An ACTION carries its text inside a \x01ACTION … \x01 frame. Unwrap
        # it so "/me prods vibebot" is matched and stored as prose, and so a
        # leading command char inside an action stays prose too.
        action_body = self._action_payload(msg)
        if action_body is not None:
            text = action_body
            if not text:
                return
        else:
            prefix_chars = conf.supybot.reply.whenAddressedBy.chars()
            if text[0] in prefix_chars:
                # Explicit command — Limnoria's dispatcher handles it.
                return

        target = msg.args[0]
        is_pm = ircutils.nickEqual(target, irc.nick)
        addressed_text = text.strip() if is_pm else self._strip_nick_address(irc.nick, text)
        if not addressed_text and action_body is not None and not is_pm:
            addressed_text = self._strip_nick_mention(irc.nick, text)
        # A bot replying to one of our lines is not talking to us: Ender
        # answers every link with a reply carrying its title.
        if (
            not addressed_text
            and not is_pm
            and self._is_reply_to_own_line(irc, msg)
            and not self._sender_is_bot(irc, msg)
        ):
            addressed_text = text.strip()

        if addressed_text:
            if action_body is not None:
                # Keep the action framing the model already speaks: the
                # "* Nick does something" form _extract_action emits.
                addressed_text = f"* {msg.nick} {addressed_text}"
            self._route_addressed_to_assistant(irc, msg, addressed_text)
            return

        # Not addressed — channel chatter only.
        channel = msg.channel
        if not channel:
            return

        if not self.registryValue("contextEnabled", channel):
            return
        if self.registryValue("privacyOptInOnly", channel):
            return

        display_nick = msg.nick
        caller = self._resolve_caller(irc, msg)
        message_text = text if action_body is None else f"* {display_nick} {action_body}"

        # Store in conversation context for richer follow-up questions
        # Use display nick for channel context (what the LLM sees) so it
        # addresses people by their visible IRC name, not their account name.
        ctx_cfg = self._get_context_config(channel)
        self.context.add_message(
            caller.identity.key, channel, Role.USER, message_text, config=ctx_cfg, persist=False
        )
        self.context.add_channel_message(
            channel, display_nick, Role.USER, message_text, config=ctx_cfg
        )

    def _will_skip_auto_who(self, irc: callbacks.Irc) -> bool:
        """Return True iff the auto-WHO on channel join should be suppressed.

        Gate: both 'account-tag' AND 'extended-join' IRCv3 caps must be ACK'd
        (account-tag rides on PRIVMSG-class messages; extended-join rides on
        JOIN itself — together they obviate the auto-WHO scan), AND the
        operator-controlled ``skipAutoWhoOnJoin`` config must be True.
        """
        if not irc_has_caps(irc, "account-tag", "extended-join"):
            return False
        return bool(self.registryValue("skipAutoWhoOnJoin"))

    @staticmethod
    def _bot_flag_key(irc: callbacks.Irc, nick: str) -> str:
        """Flag-store key. Nicks are case-insensitive; two networks are not."""
        return f"{getattr(irc, 'network', '')}:{nick.lower()}"

    def do354(self, irc: callbacks.Irc, msg: IrcMsg) -> None:  # noqa: N802
        """Record the +B flag out of a WHOX reply.

        Limnoria asks for ``%tuhnairf,1`` and unpacks the same nine fields
        (irclib.py:975) but keeps only the hostmask and account — the status
        field, where the bot flag lives, is dropped on the floor. Reading it
        here costs nothing: the reply is already arriving.
        """
        args = getattr(msg, "args", ())
        if len(args) == 9 and args[1] == _WHO_LOOKUP_TOKEN:
            # irc_lookup kind=who: <me> 7 <chan> <user> <host> <nick> <flags>
            # <account> :<realname>
            self._irc_queries.feed("who", self._network_of(irc), args[2], tuple(args[3:]))
            return
        if len(args) != 9 or args[1] != "1":
            return
        nick, status = args[5], args[6]
        with self._bot_loop_lock:
            self._bot_flags[self._bot_flag_key(irc, nick)] = "B" in status

    # ------------------------------------------------------------------
    # LIST / NAMES numerics → ircquery.IrcQueryRegistry
    # ------------------------------------------------------------------
    # How long @channels / @names / irc_lookup wait for the closing numeric.
    # A LIST on a mid-sized network completes in well under a second; the
    # margin covers a server that is throttling us.
    _IRC_QUERY_TIMEOUT = 15.0

    @staticmethod
    def _network_of(irc: callbacks.Irc) -> str:
        return str(getattr(irc, "network", "") or "")

    def do322(self, irc: callbacks.Irc, msg: IrcMsg) -> None:  # noqa: N802
        """RPL_LIST: ``<me> <channel> <count> :<topic>``."""
        args = msg.args
        if len(args) < 3:
            return
        try:
            users = int(args[2])
        except ValueError:
            users = 0
        topic = args[3] if len(args) > 3 else ""
        self._irc_queries.on_list_row(self._network_of(irc), args[1], users, topic)

    def do323(self, irc: callbacks.Irc, msg: IrcMsg) -> None:  # noqa: N802
        """RPL_LISTEND."""
        self._irc_queries.on_list_end(self._network_of(irc))

    def do353(self, irc: callbacks.Irc, msg: IrcMsg) -> None:  # noqa: N802
        """RPL_NAMREPLY: ``<me> <symbol> <channel> :<nicks>``."""
        args = msg.args
        if len(args) < 4:
            return
        self._irc_queries.on_names_row(self._network_of(irc), args[2], args[3].split())

    def do366(self, irc: callbacks.Irc, msg: IrcMsg) -> None:  # noqa: N802
        """RPL_ENDOFNAMES: ``<me> <channel> :End of /NAMES list``."""
        if len(msg.args) >= 2:
            self._irc_queries.on_names_end(self._network_of(irc), msg.args[1])

    def do403(self, irc: callbacks.Irc, msg: IrcMsg) -> None:  # noqa: N802
        """ERR_NOSUCHCHANNEL / ERR_NOSUCHNICK / RPL_TRYAGAIN: ``<me> <target> :<text>``.

        Only a query already waiting on ``<target>`` is affected; a stray
        error for something nobody asked about is dropped by the registry.
        """
        if len(msg.args) >= 3:
            self._irc_queries.on_error(self._network_of(irc), msg.args[1], msg.args[2])

    do401 = do403
    do263 = do403
    do406 = do403  # ERR_WASNOSUCHNICK
    do442 = do403  # ERR_NOTONCHANNEL (TOPIC)

    def do311(self, irc: callbacks.Irc, msg: IrcMsg) -> None:  # noqa: N802
        """WHOIS body lines: ``<me> <nick> ...`` — parsed in ircquery."""
        if len(msg.args) >= 2:
            self._irc_queries.on_whois_numeric(self._network_of(irc), msg.command, msg.args[1:])
            if msg.command in ("312", "301"):
                # WHOWAS reuses 312 (server + signoff time) and 301 (away).
                self._irc_queries.feed(
                    "whowas",
                    self._network_of(irc),
                    msg.args[1],
                    (msg.command, tuple(msg.args[1:])),
                )

    do301 = do311
    do312 = do311
    do313 = do311
    do317 = do311
    do319 = do311
    do330 = do311

    def do318(self, irc: callbacks.Irc, msg: IrcMsg) -> None:  # noqa: N802
        """RPL_ENDOFWHOIS."""
        if len(msg.args) >= 2:
            self._irc_queries.on_whois_end(self._network_of(irc), msg.args[1])

    # -- generic collectors (see IrcQueryRegistry.collect) ------------------

    def _feed_query(self, irc: callbacks.Irc, kind: str, target: str, msg: IrcMsg) -> bool:
        return self._irc_queries.feed(
            kind, self._network_of(irc), target, (msg.command, tuple(msg.args[1:]))
        )

    def do314(self, irc: callbacks.Irc, msg: IrcMsg) -> None:  # noqa: N802
        """RPL_WHOWASUSER: ``<me> <nick> <user> <host> * :<realname>``."""
        if len(msg.args) >= 2:
            self._feed_query(irc, "whowas", msg.args[1], msg)

    def do369(self, irc: callbacks.Irc, msg: IrcMsg) -> None:  # noqa: N802
        """RPL_ENDOFWHOWAS."""
        if len(msg.args) >= 2:
            self._irc_queries.close("whowas", self._network_of(irc), msg.args[1])

    def do332(self, irc: callbacks.Irc, msg: IrcMsg) -> None:  # noqa: N802
        """RPL_TOPIC: ``<me> <channel> :<topic>``."""
        if len(msg.args) >= 3:
            self._feed_query(irc, "topic", msg.args[1], msg)

    def do333(self, irc: callbacks.Irc, msg: IrcMsg) -> None:  # noqa: N802
        """RPL_TOPICWHOTIME: ``<me> <channel> <setter> <ts>`` — ends a TOPIC."""
        if len(msg.args) >= 2 and self._feed_query(irc, "topic", msg.args[1], msg):
            self._irc_queries.close("topic", self._network_of(irc), msg.args[1])

    def do331(self, irc: callbacks.Irc, msg: IrcMsg) -> None:  # noqa: N802
        """RPL_NOTOPIC."""
        if len(msg.args) >= 2:
            self._irc_queries.close(
                "topic", self._network_of(irc), msg.args[1], error="no topic is set"
            )

    def do251(self, irc: callbacks.Irc, msg: IrcMsg) -> None:  # noqa: N802
        """LUSERS / VERSION / ADMIN lines for the ``network`` lookup."""
        self._feed_query(irc, "network", "*", msg)
        if msg.command in ("259", "423"):
            # ADMIN is sent last, so its end (or "no admin info") ends it all.
            self._irc_queries.close("network", self._network_of(irc), "*")

    do252 = do251
    do253 = do251
    do254 = do251
    do255 = do251
    do256 = do251
    do257 = do251
    do258 = do251
    do259 = do251
    do265 = do251
    do266 = do251
    do351 = do251
    do423 = do251

    def _query_collect(
        self,
        irc: callbacks.Irc,
        kind: str,
        target: str,
        *lines: ircmsgs.IrcMsg,
        timeout: float | None = None,
        partial: bool = False,
    ) -> tuple[list, str | None] | None:
        """Send ``lines`` and wait for the rows a ``(kind, target)`` collects."""

        def send() -> None:
            with self._irc_send_lock:
                for line in lines:
                    irc.queueMsg(line)

        return self._irc_queries.collect(
            kind,
            self._network_of(irc),
            target,
            send,
            timeout=self._IRC_QUERY_TIMEOUT if timeout is None else timeout,
            partial=partial,
        )

    # CTCP replies come from a user's client, which may ignore them entirely.
    _CTCP_TIMEOUT = 6.0

    def _capture_ctcp_reply(self, irc: callbacks.Irc, msg: IrcMsg) -> None:
        """Hand a CTCP VERSION / PING reply NOTICE to the lookup waiting on it."""
        if len(msg.args) < 2 or not ircutils.strEqual(msg.args[0], irc.nick):
            return
        text = msg.args[1]
        for kind in ("VERSION", "PING"):
            if text.startswith(f"\x01{kind} "):
                body = text[len(kind) + 2 :].rstrip("\x01")
                net = self._network_of(irc)
                if self._irc_queries.feed(f"ctcp-{kind}", net, msg.nick or "", body):
                    self._irc_queries.close(f"ctcp-{kind}", net, msg.nick or "")
                return

    # CHATHISTORY: the server answers with a ``chathistory`` BATCH of
    # PRIVMSG / NOTICE lines. inFilter hands them to the waiting lookup and
    # drops them, so a replayed "vibebot ..." is never answered twice and
    # ChannelLogger does not log an hour of chat a second time.

    @staticmethod
    def _history_batch(irc: callbacks.Irc, msg: IrcMsg):
        """The chathistory batch ``msg`` belongs to, or None."""
        if not (getattr(msg, "server_tags", None) or {}).get("batch"):
            return None
        try:
            batches = irc.state.getParentBatches(msg)
        except (ValueError, AttributeError):
            return None
        for batch in batches:
            if batch.type == "chathistory":
                return batch
        return None

    def _capture_history(self, irc: callbacks.Irc, msg: IrcMsg) -> bool:
        """Feed a chathistory line, or close on its batch end.

        True means ``msg`` is a replayed history line and must be dropped,
        whether or not a lookup was still waiting for it: nothing in this
        plugin should act on a line from an hour ago.
        """
        net = self._network_of(irc)
        if msg.command == "BATCH":
            if msg.args and msg.args[0].startswith("-"):
                batch = msg.tagged("batch")
                if batch is not None and batch.type == "chathistory" and batch.arguments:
                    self._irc_queries.close("history", net, batch.arguments[0])
            return False
        if msg.command not in ("PRIVMSG", "NOTICE") or len(msg.args) < 2:
            return False
        batch = self._history_batch(irc, msg)
        if batch is None:
            return False
        if batch.arguments:
            tags = getattr(msg, "server_tags", None) or {}
            row = (tags.get("time", ""), msg.nick or "", msg.args[1], tags.get("msgid", ""))
            self._irc_queries.feed("history", net, batch.arguments[0], row)
        return True

    def _query_channels(self, irc: callbacks.Irc) -> list[ircquery.ChannelRow] | None:
        """Fetch (or reuse the cached) LIST for this network; None on silence."""

        def send() -> None:
            with self._irc_send_lock:
                irc.queueMsg(ircmsgs.IrcMsg(command="LIST"))

        return self._irc_queries.list_channels(
            self._network_of(irc), send, timeout=self._IRC_QUERY_TIMEOUT
        )

    def _query_names(self, irc: callbacks.Irc, channel: str) -> ircquery.NamesResult | None:
        """Ask the server who is visible in ``channel``; None on silence."""

        def send() -> None:
            with self._irc_send_lock:
                irc.queueMsg(ircmsgs.names(channel))

        return self._irc_queries.names(
            self._network_of(irc), channel, send, timeout=self._IRC_QUERY_TIMEOUT
        )

    def _query_whois(self, irc: callbacks.Irc, nick: str) -> ircquery.WhoisResult | None:
        """Remote WHOIS (``WHOIS nick nick``) so idle/signon come back; None on silence."""

        def send() -> None:
            with self._irc_send_lock:
                irc.queueMsg(ircmsgs.whois(nick, nick))

        return self._irc_queries.whois(
            self._network_of(irc), nick, send, timeout=self._IRC_QUERY_TIMEOUT
        )

    _IRC_QUERY_SILENT = "The server did not answer in time."

    def _known_bot(self, irc: callbacks.Irc, nick: str) -> bool | None:
        """True/False once the network has told us, None until it has."""
        with self._bot_loop_lock:
            return self._bot_flags.get(self._bot_flag_key(irc, nick))

    def _host_is_a_bot_host(self, prefix: object) -> bool:
        """True when the sender's host is one the network reserves for bots.

        The +B user mode is opt-in and the bot that actually loops with us does
        not set it: grok answers on ``Hxz`` from ``grok.Bot.AfterNET.Org``.
        AfterNET only hands those hosts to registered bots — a full day of
        channel logs has grok, grook5 and vibebot on one, and no humans.
        """
        if not isinstance(prefix, str) or "@" not in prefix:
            return False
        host = prefix.rsplit("@", 1)[1].lower()
        suffixes = self.registryValue("botLoopHostSuffixes")
        if isinstance(suffixes, str):
            suffixes = suffixes.split()
        elif not isinstance(suffixes, (list, tuple, set)):
            # A registry value of the wrong shape must not decide that every
            # sender is a bot, nor raise on a path every addressed line takes.
            return False
        return any(host.endswith(str(suffix).lower()) for suffix in suffixes if suffix)

    def _sender_is_bot(self, irc: callbacks.Irc, msg: IrcMsg) -> bool | None:
        """True/False/None for "is the sender a bot", cheapest signal first.

        The host is free and decides on the first line. The +B flag needs a
        WHO, so it only answers for nicks we have already asked about — None
        means nobody has told us yet and the caller should treat it as a
        person. AfterNET advertises BOT=B, so per IRCv3 bot-mode a +B sender's
        lines also carry a ``bot`` tag, which answers on the first line.
        """
        if self._host_is_a_bot_host(getattr(msg, "prefix", None)):
            return True
        tags = getattr(msg, "server_tags", None)
        if isinstance(tags, dict) and "bot" in tags:
            return True
        return self._known_bot(irc, getattr(msg, "nick", "") or "")

    def _probe_bot_flag(self, irc: callbacks.Irc, nick: str) -> None:
        """Ask the network whether ``nick`` is a bot, at most once per window.

        ``skipAutoWhoOnJoin`` is on in production and the last WHO reply the
        bot saw was six months ago, so waiting for a channel sync would mean
        never learning anything. One WHO per stranger is cheap; the answer
        lands before their second line in any real loop.
        """
        key = self._bot_flag_key(irc, nick)
        now = time.time()
        with self._bot_loop_lock:
            if key in self._bot_flags:
                return
            if now - self._bot_probed_at.get(key, 0.0) < _BOT_PROBE_INTERVAL:
                return
            self._bot_probed_at[key] = now
        irc.queueMsg(ircmsgs.who(nick, args=("%tuhnairf,1",)))

    def _note_human_turn(self, irc: callbacks.Irc, msg: IrcMsg) -> None:
        """Clear a target's bot counts when a human ADDRESSES the bot.

        Only addressed lines, which is why this hangs off the addressed route
        rather than doPrivmsg. Resetting on any human line at all reads well
        and fails badly: what people actually say during a bot loop is "wtf",
        "jesus", "but really?" — commentary ON the spam. #afternet,
        2026-08-24 21:47-21:49 carried six such lines in two minutes, and a
        reset on each would have licensed three more replies apiece, roughly
        eighteen in all. A person talking TO the bot is a conversation; a
        person talking about the noise is the noise being a problem.
        """
        nick = getattr(msg, "nick", "")
        target = msg.args[0] if msg.args else ""
        if not nick or not target:
            return
        if self._sender_is_bot(irc, msg):
            return
        with self._bot_loop_lock:
            self._bot_reply_counts = {
                key: value for key, value in self._bot_reply_counts.items() if key[0] != target
            }

    def _bot_loop_blocked(self, irc: callbacks.Irc, msg: IrcMsg) -> bool:
        """True when this line is one robot talking to another, too many times.

        Unknown nicks are treated as people: the flag store fails open, so the
        worst case is the loop we already have rather than a silenced user. A
        probe is fired on the way past so the next line is decidable.
        """
        nick = getattr(msg, "nick", "")
        target = msg.args[0] if msg.args else ""
        if not nick or not target:
            return False

        flag = self._sender_is_bot(irc, msg)
        if flag is None:
            self._probe_bot_flag(irc, nick)
            return False
        if not flag:
            return False

        channel = msg.channel or None
        limit = self.registryValue("botLoopReplyLimit", channel)
        if not limit:
            return False
        window = self.registryValue("botLoopWindow", channel)

        now = time.time()
        key = (target, nick.lower())
        with self._bot_loop_lock:
            count, last = self._bot_reply_counts.get(key, (0, 0.0))
            if now - last > window:
                count = 0
            if count >= limit:
                blocked = True
            else:
                self._bot_reply_counts[key] = (count + 1, now)
                blocked = False
        if blocked:
            self.log.info(
                "bot_loop_capped nick=%s target=%s limit=%i window=%is",
                nick,
                target,
                limit,
                window,
            )
        return blocked

    def doTagmsg(self, irc: callbacks.Irc, msg: IrcMsg) -> None:  # noqa: N802
        """Capture inbound IRCv3 emoji reactions (+draft/react) to verse lines.

        Recency-attributed, measurement only (no reply). Fully exception-isolated
        so a capture bug can never disturb the IRC event loop. See verse/reactions.py.
        """
        try:
            server_tags = getattr(msg, "server_tags", None) or {}
            react_emoji = server_tags.get("+draft/react")
            if not react_emoji:
                return
            if ircutils.strEqual(msg.nick, irc.nick):  # never count the bot's own reactions
                return
            channel = msg.channel or (msg.args[0] if msg.args else "")
            if not channel or not channel.startswith(("#", "&")):
                return
            if not self.registryValue("verseReactionCaptureEnabled", channel):
                return
            with self._irc_send_lock:
                last = self._last_bot_line.get((irc.network, ircutils.toLower(channel)))
            event = reactions.process_reaction(
                react_emoji=react_emoji,
                reactor=msg.nick,
                channel=channel,
                network=irc.network,
                target_msgid=server_tags.get("+draft/reply"),
                last_bot_line=last,
                now=time.time(),
                capture_enabled=True,
            )
            if event is not None:
                self._append_reaction_event(event)
        except Exception:
            self.log.exception("doTagmsg reaction capture failed")

    def _append_reaction_event(self, event: dict) -> None:
        """Append one reaction event to <data>/verse/reactions.jsonl (thread-safe)."""
        base = Path(conf.supybot.directories.data()) / "verse"
        path = base / "reactions.jsonl"
        with self._reaction_log_lock:
            base.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(reactions.event_to_jsonl(event) + "\n")

    def doJoin(self, irc: callbacks.Irc, msg: IrcMsg) -> None:  # noqa: N802
        """Track channels the bot is joining for startup notification.

        When the bot joins a channel, we add it to _pending_channels.
        The channel is removed when we receive do315 (end of WHO).

        If the auto-WHO on join is being suppressed (account-tag + extended-join
        + skipAutoWhoOnJoin), do315 will never fire — so we must NOT add to
        _pending_channels here. The do376 2-second fallback (line 828) is then
        responsible for firing the startup notification.
        """
        if not ircutils.strEqual(irc.nick, msg.nick):
            return
        if self._will_skip_auto_who(irc):
            return
        channel = msg.args[0]
        self._pending_channels.add(channel)

    def do315(self, irc: callbacks.Irc, msg: IrcMsg) -> None:  # noqa: N802
        """Handle end of WHO reply (channel sync complete).

        When a channel finishes syncing (WHO complete), remove it from
        pending channels. When all channels are synced and we haven't
        notified yet, send the startup notification.
        """
        channel = msg.args[1]
        self._irc_queries.close("who", self._network_of(irc), channel)
        self._pending_channels.discard(channel)

        if not self._pending_channels and not self._startup_notified:
            self._send_startup_notification(irc)
            self._startup_notified = True

    def do376(self, irc: callbacks.Irc, msg: IrcMsg) -> None:  # noqa: N802
        """Handle end of MOTD (connection established).

        Reset startup tracking state on reconnection so we send a fresh
        notification. Also handles case where bot has no channels configured.
        """
        self._pending_channels.clear()
        self._startup_notified = False
        if hasattr(self, "db"):
            self._resend_monitors(irc)

        # If no channels are configured, send notification immediately
        # (we need to check after a short delay to allow channel joins to start)
        def check_no_channels() -> None:
            if not self._pending_channels and not self._startup_notified:
                self._send_startup_notification(irc)
                self._startup_notified = True

        # Schedule check after 2 seconds to allow join commands to be processed
        with contextlib.suppress(KeyError):
            schedule.removeEvent("llm_startup_check")
        schedule.addEvent(check_no_channels, time.time() + 2, name="llm_startup_check")

    def _send_startup_notification(self, irc: callbacks.Irc) -> None:
        """Send startup notification PM to bot owner.

        Message format: VibeBot started | v8 | N channel(s) | YYYY-MM-DD HH:MM:SS UTC
        """
        # Remove the scheduled check event if it exists
        with contextlib.suppress(KeyError):
            schedule.removeEvent("llm_startup_check")

        # Find users with owner capability.
        users_mod = getattr(ircdb, "users", None)
        users_map = getattr(users_mod, "users", {})
        owners = [user.name for user in users_map.values() if "owner" in user.capabilities]
        if not owners:
            self.log.warning("No bot owner configured, skipping startup notification")
            return

        owner = owners[0]
        channel_count = len(irc.state.channels)
        timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        plural = "s" if channel_count != 1 else ""
        message = f"VibeBot started | v8 | {channel_count} channel{plural} | {timestamp}"

        irc.queueMsg(ircmsgs.privmsg(owner, message))
        self.log.info("Startup notification sent to %s", owner)

    @staticmethod
    def _get_build_info() -> str:
        """Get version and git commit SHA for context prompt.

        Returns:
            Build string like "v0.1.0 (abc1234)" or just "v0.1.0" if git unavailable.
        """
        from . import __version__

        try:
            sha = subprocess.check_output(  # noqa: S603
                ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            return f"v{__version__} ({sha})"
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            return f"v{__version__}"

    def _init_context(self) -> None:
        """Initialize context manager with global defaults (called once at startup).

        Per-channel overrides are read at query time via ``_get_context_config``.
        """
        config = ContextConfig(
            max_messages=self.registryValue("contextMaxMessages"),
            timeout_minutes=self.registryValue("contextTimeoutMinutes"),
            enabled=self.registryValue("contextEnabled"),
            channel_max_messages=self.registryValue("channelContextMaxMessages"),
        )
        self.context = ConversationContext(config, db=self.db)

    def _get_context_config(self, channel: str) -> ContextConfig:
        """Read channel-specific context configuration.

        Args:
            channel: IRC channel name (passed to ``registryValue``
                for per-channel overrides)

        Returns:
            ContextConfig with channel-specific values
        """
        return ContextConfig(
            max_messages=self.registryValue("contextMaxMessages", channel),
            timeout_minutes=self.registryValue("contextTimeoutMinutes", channel),
            enabled=self.registryValue("contextEnabled", channel),
            channel_max_messages=self.registryValue("channelContextMaxMessages", channel),
        )

    def _send_reminder_text(
        self,
        irc: callbacks.Irc,
        target: str,
        nick: str,
        text: str,
        *,
        label: str = "",
        reply_msgid: str = "",
    ) -> None:
        """Match the inner _send closure semantics for reminder output.

        Reminders bypass _send_long_reply, so collapse multi-line content
        here (raw \\n in a PRIVMSG body causes Excess Flood disconnects
        on AfterNET) and pastebin anything longer than one wire line.

        ``reply_msgid`` threads the fire under the message that set the
        reminder via ``+draft/reply``, matching draw delivery. When that tag
        actually goes on the wire the ``Reminder (label):`` wrapper is
        dropped — the client already shows the request, so restating it is
        noise. When the server never negotiated ``message-tags`` the wrapper
        stays, because an unattached line has to say what it is answering.
        """
        reply_to = reply_msgid if reply_msgid and irc_has_caps(irc, "message-tags") else None
        if not reply_to:
            text = f"Reminder ({label}): {text}" if label else f"Reminder: {text}"
        safe_text = self.llm_service.sanitize_output(text)
        line = self._format_pending_completed_reply(nick, safe_text, target)
        self._safe_queue(irc, self._safe_privmsg(target, line, reply_to))

    _AGENDA_WAKEUP_EVENT = "llm_agenda_wakeup"

    def _arm_agenda(self) -> None:
        """Point the one agenda wakeup at the next reminder or task due.

        Called after anything that adds, fires, or settles an entry. Only
        ever moves the wakeup earlier: a wakeup that turns out early (its
        entry was cancelled) finds nothing due and re-arms. Serialised,
        because request threads and fire workers both arm it, and an
        interleaved remove/add can lose the earlier wakeup or trip
        Limnoria's duplicate-name assert (the queue hit this, a5ea4a9).
        """
        try:
            with self._agenda_wakeup_lock:
                self._arm_agenda_locked()
        except Exception:
            # The rows are untouched, so the next arm (any new entry, any
            # settled fire, a restart) still finds them.
            self.log.exception("agenda_arm_failed")

    def _arm_agenda_locked(self) -> None:
        at = self.agenda.next_due_at()
        if at is None:
            return
        now = time.time()
        # Limnoria drops events in the past; an overdue entry fires next tick.
        effective = max(at, now + 1)
        current = self._agenda_wakeup_at
        if current is not None and now < current <= effective:
            return
        with contextlib.suppress(KeyError):
            schedule.removeEvent(self._AGENDA_WAKEUP_EVENT)
        schedule.addEvent(self._agenda_tick, effective, name=self._AGENDA_WAKEUP_EVENT)
        self._agenda_wakeup_at = effective

    def _agenda_tick(self) -> None:
        """The agenda wakeup: fire everything due, then re-arm.

        Runs on the scheduler thread. Each fire hands slow work to a worker;
        whoever finishes a fire calls :meth:`_finish_agenda`.
        """
        with self._agenda_wakeup_lock:
            self._agenda_wakeup_at = None
        if self._llm_executor.closing:
            return
        try:
            claim = self.agenda.claim_due()
        except Exception:
            self.log.exception("agenda_claim_failed")
            claim = agenda.Claim(due=[], expired=[])
        for gone in claim.expired:
            self.log.info(
                "agenda_expired kind=%s event=%s fire_at=%.3f",
                gone.kind,
                gone.row.event_name,
                gone.row.fire_at,
            )
        for due in claim.due:
            fired_at = time.time()
            try:
                if isinstance(due.row, ReminderRow):
                    self._fire_reminder(due.row, fired_at)
                else:
                    self.llm_service.fire_scheduled_llm_task(due.row, fired_at)
            except Exception:
                self.log.exception(
                    "agenda_fire_failed kind=%s event=%s", due.kind, due.row.event_name
                )
                self._finish_agenda(due.kind, due.row.event_name, fired_at)
        self._arm_agenda()

    def _finish_agenda(self, kind: agenda.Kind, event_name: str, fired_at: float) -> None:
        """Settle a fired entry (reschedule it in place, or remove it) and re-arm.

        Every fire path ends here. A second call for the same fire is a no-op
        in the agenda, so it cannot advance a chain twice. Skipped while
        shutting down, so a dying plugin does not write to the database; the
        row is still there and fires again on the next start.
        """
        if self._llm_executor.closing:
            return
        try:
            done = self.agenda.finish(kind, event_name, fired_at=fired_at)
        except Exception:
            self.log.exception("agenda_finish_failed kind=%s event=%s", kind, event_name)
            self._arm_agenda()
            return
        if done.outcome is agenda.Outcome.RESCHEDULED:
            self.log.info(
                "agenda_reschedule kind=%s event=%s position=%s/%s next_fire_at=%.3f",
                kind,
                event_name,
                done.chain_position,
                done.cap,
                done.next_fire_at,
            )
        elif done.outcome is not agenda.Outcome.DONE:
            self.log.info(
                "agenda_reschedule_skipped kind=%s event=%s reason=%s",
                kind,
                event_name,
                done.outcome,
            )
        self._arm_agenda()

    def _unattended_ask_rate_limited(self, *, account: str | None, nick: str, now: float) -> bool:
        """Silent daily-ask limit PEEK for fires that bypass command preflight.

        ``silent=True`` suppresses the user-facing error and the
        ``rate_limited`` usage row — callers deliver their own skip notice.
        ``record=False`` means the fire is skipped when the user has maxed
        their own interactive use but never consumes an interactive slot
        itself. Shared by reminder action fires and scheduled LLM tasks.
        """
        rl_account = account if account else nick
        rl_tier = "registered" if account else "unregistered"
        return self._check_rate_limit(
            None,
            "ask",
            rl_account,
            "",
            "",
            "",
            tier=rl_tier,
            silent=True,
            record=False,
            now=now,
        )

    def _run_unattended_assistant(
        self,
        *,
        irc: callbacks.Irc,
        msg: IrcMsg,
        prompt: str,
        caller: Caller,
        bot_nick: str,
        entry_route: str,
        exclude_tools: frozenset[str] = frozenset(),
        fold_instruction_into_prompt: bool = False,
    ) -> AssistantResult:
        """Fire an LLM action outside the normal command preflight.

        Shared plumbing for reminder action fires (``_fire_reminder_action``)
        and scheduled LLM tasks (``service._dispatch_scheduled_task``): both
        dispatch ``assistant_completion`` from a worker thread with no live
        command context, so the synthetic request context, the
        history/memory/instruction gathering, and the standard tool-callback
        wiring live here. Callers keep their own rate-limit/capability
        policy, delivery, usage logging, and reschedule logic.

        ``fold_instruction_into_prompt`` preserves the scheduled-task
        variant, which folds the user's @instruct into the system prompt
        (and passes no channel overlay when there is no instruction). The
        reminder path instead rides the @instruct as user-role data (see
        assistant_completion ``user_instruction``) and always passes the overlay.
        """
        request_context = AssistantRequestContext(
            entry_route=entry_route,
            profile=PROFILE_REMIND_ACTION,
            caller=caller,
            is_owner=False,
            # Same per-feature caps as @ask/@draw; owner/admin excluded.
            capabilities=frozenset({"llm.ask", "llm.draw"}),
        )
        channel = caller.reply_target
        history, channel_history = self._gather_history(caller)
        personal_key = caller.identity.personal_key
        memories = self._get_user_memories(personal_key) if personal_key else []
        user_instruction = self.db.get_instruction(personal_key) if personal_key else None
        ask_prompt = self.registryValue(PROFILES[PROFILE_REMIND_ACTION].overlay_setting, channel)
        if fold_instruction_into_prompt:
            system_prompt = f"{user_instruction}\n\n{ask_prompt}" if user_instruction else None
            user_instruction = None
        else:
            system_prompt = ask_prompt
        return self.llm_service.assistant_completion(
            prompt=prompt,
            **request_context.completion_kwargs(),
            db=self.db,
            context=self.context,
            bot_nick=bot_nick,
            history=history,
            channel_history=channel_history,
            irc=irc,
            msg=msg,
            memories=memories,
            user_instruction=user_instruction,
            system_prompt=system_prompt,
            search_fn=lambda q: self.llm_service.search_completion(q, channel=channel),
            fetch_fn=lambda u: self.llm_service.url_completion(u, channel=channel),
            draw_fn=lambda p, _irc=irc, _msg=msg, **kw: self._draw_for_assistant(
                _irc, _msg, p, **kw
            ),
            cleanup_fn=lambda n: self._run_memory_cleanup(n, channel),
            exclude_tools=exclude_tools,
            **self._pending_task_fns(
                caller=caller,
                irc=irc,
                msg=msg,
                pass_irc_msg_to_callbacks=False,
            ),
        )

    def _log_unattended_usage(
        self,
        *,
        nick: str,
        account: str | None,
        channel: str,
        command: str,
        prompt: str,
        result: AssistantResult,
        silent: bool,
        log_context: str,
    ) -> None:
        """Best-effort usage logging for fires that bypass command preflight.

        Attributes the fire's LLM cost to the chain owner (account, falling
        back to nick). Never raises — a logging failure must not break
        delivery or rescheduling. An empty reply counts as undelivered, the
        same as on the command path (see ``_usage_status``).
        """
        if silent:
            status, error_detail = "silent", (result.error or "")[:200]
        else:
            delivered = bool((result.content or "").strip())
            status, error_detail = self._usage_status(result, delivered=delivered)
        try:
            self.db.log_usage(
                Identity(nick, account).key,
                channel,
                command,
                result.model,
                result.prompt_tokens,
                result.completion_tokens,
                result.cost,
                prompt=prompt,
                status=status,
                error_detail=error_detail,
            )
        except Exception:
            self.log.exception("%s usage log failed: %s", command, log_context)

    def _fire_reminder_action(
        self,
        row: ReminderRow,
        *,
        active_irc: callbacks.Irc,
        target: str,
        bot_nick: str,
        fired_at: float,
    ) -> None:
        """Worker-thread reminder action body.

        Resolves history, performs the LLM call, dispatches the response,
        logs usage, and settles the fire with the agenda (which reschedules a
        recurring row in place). Settling always runs in the worker's finally
        regardless of inner-try outcome — a single failed fire does not kill
        a recurring watch chain.
        """
        nick = row.nick
        account = row.account
        channel = row.channel
        message = row.message
        event_name = row.event_name
        action_prompt = row.action_prompt
        reply_msgid = row.reply_msgid
        try:
            msg_target = Caller.from_row(nick, account, channel).reply_target
            msg_kwargs: dict[str, object] = {
                "prefix": f"{nick}!~remind@scheduled",
                "command": "PRIVMSG",
                "args": (msg_target, ""),
            }
            if account:
                msg_kwargs["server_tags"] = {"account": account}
            synthetic_msg = ircmsgs.IrcMsg(**msg_kwargs)

            # The user's @instruct rides as user-role data (see assistant_completion
            # user_instruction), not prepended to the system overlay.
            result = self._run_unattended_assistant(
                irc=active_irc,
                msg=synthetic_msg,
                # Without the frame, "check if you reminded me" reads as a
                # request to audit the pending list (ibutsu, 2026-09-25).
                prompt=(
                    f"({nick}'s reminder, set earlier, is firing now: this "
                    "reply IS the reminder being delivered, so it is no "
                    f"longer pending.)\n\n{action_prompt}"
                ),
                caller=Caller.from_row(nick, account, channel),
                bot_nick=bot_nick,
                entry_route=PROFILE_REMIND_ACTION,
                exclude_tools=frozenset({"set_reminder"}),
            )
            response = result.content.strip() if result.content else ""
            is_silent = response == "[silent]"
            # Booked before delivery, so a send that raises cannot lose the
            # spend. Attributes the action fire's LLM cost to the chain owner.
            self._log_unattended_usage(
                nick=nick,
                account=account,
                channel=channel,
                command="remind_action",
                prompt=action_prompt,
                result=result,
                silent=is_silent,
                log_context=f"event={event_name}",
            )
            if is_silent:
                pass  # No user-visible output this fire.
            elif not response:
                self._send_reminder_text(
                    active_irc,
                    target,
                    nick,
                    f"{message} (action returned empty response)",
                    reply_msgid=reply_msgid,
                )
            else:
                self._send_reminder_text(
                    active_irc,
                    target,
                    nick,
                    response,
                    label=message,
                    reply_msgid=reply_msgid,
                )
        except Exception:
            self.log.exception(
                "reminder_action_delivery_failed nick=%s channel=%s event=%s",
                nick,
                channel,
                event_name,
            )
            try:
                self._send_reminder_text(
                    active_irc,
                    target,
                    nick,
                    f"Action '{message}' failed. (Set this reminder again to retry.)",
                    reply_msgid=reply_msgid,
                )
            except Exception:
                self.log.exception(
                    "reminder_action_fallback_failed nick=%s channel=%s event=%s",
                    nick,
                    channel,
                    event_name,
                )
        finally:
            # Runs even if the inner try raised — recurring chains survive a
            # single failed fire.
            self._finish_agenda(agenda.Kind.REMINDER, event_name, fired_at)

    def _fire_reminder(self, row: ReminderRow, fired_at: float) -> None:
        """Fire a due reminder: runs on the scheduler thread, from the agenda wakeup.

        Resolves the active IRC connection, runs the rate-limit precheck,
        then either submits the heavy LLM-call body to the executor (for
        action-prompt reminders) or sends the plain-echo reminder directly.
        The fire is settled with the agenda here for every branch that does
        not reach a worker; the worker settles its own.
        """
        nick = row.nick
        account = row.account
        message = row.message
        event_name = row.event_name
        reply_msgid = row.reply_msgid
        # A PM reminder stores a nick as its channel (the bot's own, in rows
        # set before the reply target was stored); deliver to the user.
        target = Caller.from_row(nick, account, row.channel).reply_target
        submitted = False
        try:
            active_irc = next(iter(world.ircs), None)
            if active_irc is None:
                return

            # Plain echo delivery.
            if not row.action_prompt:
                self._send_reminder_text(active_irc, target, nick, message, reply_msgid=reply_msgid)
                return

            bot_nick = getattr(active_irc, "nick", None)
            if not bot_nick:
                self.log.warning(
                    "reminder_action_missing_bot_nick nick=%s channel=%s event=%s",
                    nick,
                    row.channel,
                    event_name,
                )
                self._send_reminder_text(active_irc, target, nick, message, reply_msgid=reply_msgid)
                return

            if self._unattended_ask_rate_limited(account=account, nick=nick, now=fired_at):
                self._send_reminder_text(
                    active_irc,
                    target,
                    nick,
                    f"{message} (action skipped — daily ask limit reached)",
                    reply_msgid=reply_msgid,
                )
                return

            if self._llm_executor.closing:
                return

            self._llm_executor.submit(
                f"reminder:{event_name}",
                self._fire_reminder_action,
                row,
                active_irc=active_irc,
                target=target,
                bot_nick=bot_nick,
                fired_at=fired_at,
            )
            submitted = True
        finally:
            if not submitted:
                self._finish_agenda(agenda.Kind.REMINDER, event_name, fired_at)

    @contextlib.contextmanager
    def _allow_concurrent(self):
        """Temporarily release the MetaSynchronized RLock for concurrent commands.

        Limnoria's Commands base class wraps callCommand() with an RLock,
        which serializes all command execution per-plugin. This releases the
        lock around blocking I/O (LLM API calls) so multiple commands can
        run concurrently.

        Uses RLock._release_save()/_acquire_restore() — the same mechanism
        threading.Condition uses internally.

        WARNING: These are private CPython implementation details (prefixed
        with ``_``) and are not guaranteed by the Python language spec. They
        are stable in CPython 3.12-3.14 and used by threading.Condition, but
        could break on alternative interpreters or future CPython versions.
        """
        lock = self._MetaSynchronized_rlock
        try:
            saved = lock._release_save()
        except RuntimeError:
            # Lock not held (e.g., direct call in tests) — just proceed
            yield
            return
        try:
            yield
        finally:
            lock._acquire_restore(saved)

    @contextlib.contextmanager
    def _trace_request(self, command: str, nick: str, channel: str):
        """Set a unique trace ID for the duration of a command invocation.

        All log messages emitted while the context manager is active
        will be prefixed with [trace_id] by TraceFilter.
        """
        rid = generate_request_id()
        token = request_id.set(rid)
        self.log.info("%s from %s/%s", command, channel, nick)
        try:
            yield rid
        finally:
            self.log.info("%s complete: %s/%s", command, channel, nick)
            request_id.reset(token)

    def getPluginHelp(self) -> str:  # noqa: N802
        """Return plugin help with documentation URL."""
        url = self.registryValue("helpUrl")
        names = ", ".join(cmd.name for cmd in COMMAND_REGISTRY)
        return _("AI-powered commands. Commands: %s. Full documentation: %s") % (
            names,
            url,
        )

    def invalidCommand(  # noqa: N802
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        tokens: list[str],
    ) -> None:
        """Route unrecognized addressed text through the chat profile.

        When someone says "vibebot draw a cat" or "vibebot what time is it"
        without a command prefix, the chat profile handles everything — general
        questions AND tool-based operations — via ``_ask_impl`` with the
        ``assistant_completion`` tool loop.
        """
        if not tokens:
            return

        # Check if user has ask capability
        if not ircdb.checkCapability(msg.prefix, "llm.ask"):
            return

        # Skip ZNC playback messages
        if self._is_old_message(msg):
            return

        text = " ".join(tokens)
        preflight = self._run_preflight(irc, msg, text, "ask", require_account=False)
        if preflight.blocked:
            return

        self._dispatch_addressed_async(irc, msg, text, preflight, entry_route="invalid_command")

    @staticmethod
    def _strip_nick_address(bot_nick: str, text: str) -> str | None:
        """Return ``text`` with a bot-nick address removed, or None if unaddressed.

        Matches Limnoria's nick-addressing rules at BOTH ends of the line:

        - leading (``supybot.reply.whenAddressedBy.nick``) — ``vibebot: hi``,
          ``vibebot, hi``, ``vibebot hi`` all yield ``hi``.
        - trailing (``supybot.reply.whenAddressedBy.nick.atEnd``) — ``hi,
          vibebot``, ``prods vibebot``, ``what time is it vibebot?``. Sentence
          punctuation after the nick is allowed; Limnoria's own matcher stops
          at it.

        The nick must sit on a non-alnum boundary at both ends, so neither
        ``vibebotter`` (leading) nor ``reboot`` for nick ``bot`` (trailing) is
        an address. A mention with running text on both sides — "I asked
        vibebot about it" — is deliberately NOT an address: third-person talk
        about the bot should not summon it.
        """
        stripped = text.strip()
        if not stripped:
            return None
        nick_len = len(bot_nick)

        if ircutils.strEqual(stripped[:nick_len], bot_nick):
            rest = stripped[nick_len:]
            if not rest:
                return None
            if rest[0] in _ADDRESS_SEPARATORS:
                return rest.lstrip(_ADDRESS_SEPARATORS).strip() or None

        head = stripped.rstrip(_ADDRESS_SEPARATORS + ".!?")
        if not ircutils.strEqual(head[-nick_len:], bot_nick):
            return None
        rest = head[:-nick_len]
        if not rest or rest[-1] not in _ADDRESS_SEPARATORS:
            return None
        return rest.rstrip(_ADDRESS_SEPARATORS).strip() or None

    @staticmethod
    def _strip_nick_mention(bot_nick: str, text: str) -> str | None:
        """Return ``text`` with a mid-line bot-nick token cut out, or ``None``.

        The ACTION-only counterpart to the NickInMiddle plugin. That plugin
        skips CTCP on purpose: it addresses the bot by shifting the nick to the
        front, which would turn "asks vibebot for a beer" into "vibebot asks
        for a beer" and invert who is asking. Here the nick is cut in place,
        leaving "asks for a beer".

        Plain messages are left alone — NickInMiddle owns that rewrite and is
        switchable per channel, and matching it here too would defeat the
        switch.
        """
        for match in _NICK_TOKEN_RE.finditer(text):
            start, end = match.span()
            if start == 0 or end == len(text):
                continue
            if text[start - 1] not in _ADDRESS_SEPARATORS or text[end] not in _ADDRESS_SEPARATORS:
                continue
            if not ircutils.nickEqual(match.group(0).rstrip("?.!"), bot_nick):
                continue
            before = text[:start].rstrip(_ADDRESS_SEPARATORS)
            after = text[end:].lstrip(_ADDRESS_SEPARATORS)
            if before and after:
                return f"{before} {after}"
        return None

    @staticmethod
    def _action_payload(msg: IrcMsg) -> str | None:
        """Return the text of a CTCP ACTION (``/me``) PRIVMSG, else ``None``.

        Read straight off the wire text rather than via ``ircmsgs.unAction``,
        which asserts ``isAction`` and then raises ``AttributeError`` on the
        degenerate empty action ``\x01ACTION\x01``.
        """
        text = msg.args[1] if len(msg.args) > 1 else ""
        match = _ACTION_RE.match(text)
        return match.group(1) if match else None

    def _route_addressed_to_assistant(self, irc: callbacks.Irc, msg: IrcMsg, text: str) -> None:
        """Run preflight and dispatch addressed text through ``_ask_impl``."""
        if not ircdb.checkCapability(msg.prefix, "llm.ask"):
            return
        # Bot-loop guard. Wrapped because it runs on every addressed line and
        # a guard that can swallow a reply is worse than the loop it prevents.
        try:
            if self._bot_loop_blocked(irc, msg):
                return
            self._note_human_turn(irc, msg)
        except Exception:
            self.log.exception("bot loop guard failed nick=%s", getattr(msg, "nick", ""))
        preflight = self._run_preflight(irc, msg, text, "ask", require_account=False)
        if preflight.blocked:
            return
        self._dispatch_addressed_async(irc, msg, text, preflight, entry_route="addressed")

    def _is_duplicate_dispatch(self, channel: str, text: str) -> bool:
        """True if identical addressed text hit ``channel`` within the dedup
        window — a relay echo (LarryBot re-injecting a "vibebot …" line) or a
        double entry-point that would otherwise spawn a second answer, or a
        second illustrated story. Records the sighting and prunes stale keys.
        """
        key = (channel or "", " ".join((text or "").split()).lower())
        now = time.time()
        with self._recent_dispatch_lock:
            for k in [
                k for k, ts in self._recent_dispatch.items() if now - ts > _DISPATCH_DEDUP_WINDOW
            ]:
                del self._recent_dispatch[k]
            prev = self._recent_dispatch.get(key)
            self._recent_dispatch[key] = now
            return prev is not None and (now - prev) < _DISPATCH_DEDUP_WINDOW

    def _dispatch_addressed_async(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        text: str,
        preflight: PreflightResult,
        *,
        entry_route: str,
        force_roleplay: bool = False,
    ) -> None:
        """Run addressed-text dispatch in a daemon thread so the IRC driver
        thread is freed to flush the typing indicator immediately.

        ``doPrivmsg`` (nick-addressed text and PMs) and ``invalidCommand``
        (bare ``vibebot foo`` that isn't a known command) are ``do*`` /
        invalid-command callbacks that Limnoria runs **synchronously** inside
        ``Irc.feedMsg`` on the socket driver's thread. That driver only
        flushes its outbound queue *after* the callback returns:
        ``supybot.drivers.Socket._read`` loops ``feedMsg`` over every received
        line, then calls ``_sendIfMsgs`` once at the end. Running the
        multi-second LLM generation inline therefore pins the driver — the
        ``+typing=active`` TAGMSG that ``_begin_typing`` enqueues cannot leave
        the socket until generation finishes, so "is composing" only shows up
        at the same instant as the reply. (Moving ``_begin_typing`` earlier in
        the call chain — already done — can't fix this; the bottleneck is the
        blocked flush, not call ordering.)

        Explicit ``@``-prefixed commands don't lag because Limnoria already
        runs them in a ``CommandThread`` (see ``callbacks.py``), which frees
        the driver to flush right away — that's why ``@ask`` shows typing
        promptly but nick-addressing doesn't. Offloading here mirrors that: a
        daemon ``SupyThread`` runs the dispatch, the callback returns at once,
        and the driver flushes the typing TAGMSG on its next loop iteration.
        LLM concurrency stays bounded by the ``_llm_executor.permit()`` inside
        ``_ask_impl`` (identical to the command path), so this unblocks the
        driver without adding new concurrency.

        Deliberately NOT routed through ``_llm_executor.submit``: that worker
        acquires the concurrency semaphore itself, and ``_ask_impl`` acquires
        it again via ``permit()``, so a submitted dispatch would double-acquire
        and deadlock once the pool fills — ``submit`` guards against exactly
        this nesting with ``RecursiveSubmitError``.
        """
        # Suppress a duplicate of the same addressed line (relay echo / double
        # entry-point) before spending an LLM turn or spawning a second story.
        # Bucket by reply target: two people's PMs never share one.
        if self._is_duplicate_dispatch(preflight.caller.reply_target, text):
            return

        def _work() -> None:
            # Shutdown may have begun between spawn and execution. Bail
            # before any DB/LLM work so we don't read a database that die()
            # is tearing down: die() flips the executor's closing flag first
            # thing, so it doubles as the "shutdown in progress" signal for
            # these daemon threads, which are untracked and not awaited by
            # _llm_executor.drain().
            if self._llm_executor.closing:
                return
            try:
                self._dispatch_with_verse_routing(
                    irc,
                    msg,
                    text,
                    preflight,
                    entry_route=entry_route,
                    force_roleplay=force_roleplay,
                )
            except Exception:
                self.log.exception("addressed dispatch failed in worker thread")

        world.SupyThread(
            target=_work,
            name=f"llm-addressed-{entry_route}",
            daemon=True,
        ).start()

    def _account_from_msg(self, irc: callbacks.Irc, msg: IrcMsg) -> str | None:
        """Resolve the requesting user's account name from an incoming message.

        Two layers, in order:
        1. ``msg.server_tags['account']`` via :func:`account_from_server_tags`
           — the IRCv3 ``account-tag`` capability. Rides on every
           PRIVMSG/NOTICE/TAGMSG from an identified user, so it's valid even
           for users idling in-channel since before bot start.
        2. ``irc.state.nickToAccount(nick)`` — Limnoria's session cache.
           Populated by account-tag ingest, account-notify, extended-join,
           and WHO replies.

        Returns ``None`` when the user is not identified or unknown.

        Note: this resolver does NOT consult ``ircdb`` hostmask matching. That
        path would silently promote unidentified users to the ``registered``
        tier; owner/admin/trusted gating uses ``ircdb.checkCapability(prefix, …)``
        separately and is unaffected.
        """
        tag_account = account_from_server_tags(msg)
        if tag_account:
            return tag_account
        # Server-originated messages (e.g. some operator NOTICEs) carry a
        # bare server prefix instead of nick!user@host. ``nickFromHostmask``
        # asserts user-hostmask form, so guard before calling.
        if not msg.prefix or not ircutils.isUserHostmask(msg.prefix):
            return None
        nick = ircutils.nickFromHostmask(msg.prefix)
        try:
            return irc.state.nickToAccount(nick)
        except (KeyError, AttributeError):
            return None

    def _resolve_nick_to_identity(self, irc: callbacks.Irc, nick: str) -> str:
        """Resolve a plain nick to the storage key its usage and history live under.

        AfterNet supports ``account-notify``, so Limnoria caches account names
        in ``irc.state.nicksToAccounts``. A nick authenticated to an account
        resolves to the account. A nick the cache knows to be logged out
        resolves to its unauthenticated storage key (``!nick``), so a
        squatter's rows are never read as the account's. A nick the cache
        hasn't seen (offline, or no shared channel) is taken as typed: an
        operator looking up someone who isn't online names their account.

        Args:
            irc: IRC connection (provides account lookup via ``state``)
            nick: Plain IRC nick (no hostmask)

        Returns:
            The account, ``!nick`` for a logged-out nick, or ``nick`` unchanged
            when the cache has no answer.
        """
        try:
            account = irc.state.nickToAccount(nick)
        except (KeyError, AttributeError):
            return nick
        return Identity(nick, account or None).key

    def _book_pending_usage(self, task_id: int, *, status: str, error_detail: str = "") -> None:
        """Book a pending task's billed retry at one of its end states.

        Exactly once per task (``LLMDatabase.book_pending_task_usage``), so
        it is safe to call on every path that ends a task. Never raises: an
        accounting write must not sink a delivery or its ack.
        """
        try:
            self.db.book_pending_task_usage(task_id, status=status, error_detail=error_detail)
        except Exception:
            self.log.exception("pending task usage booking failed: task_id=%s", task_id)

    def _extract_action(self, irc: callbacks.Irc, response: str) -> str | None:
        """Return action text if *response* looks like an IRC action, else ``None``.

        Recognises both ``/me does something`` and ``* BotNick does something``.
        Embedded newlines are collapsed because IRC ACTION payloads are
        single-line; sending a multi-line action would put raw ``\\n`` bytes on
        the wire, which the server parses as separate commands and treats as
        Excess Flood.
        """
        if response.startswith("/me ") and len(response) > 4:
            return self._collapse_for_irc(response[4:]) or None
        star_prefix = f"* {irc.nick} "
        if response.startswith(star_prefix) and len(response) > len(star_prefix):
            return self._collapse_for_irc(response[len(star_prefix) :]) or None
        return None

    @staticmethod
    def _collapse_for_irc(text: str) -> str:
        """Collapse multi-line text into a single IRC-safe line."""
        return " | ".join(line for line in text.splitlines() if line.strip())

    def _reply_mores_length(self, target: str | None, network: str | None = None) -> int:
        """The wire-line budget (``supybot.reply.mores.length``), scoped.

        Read the same way in three places — this delivery formatter, the
        pending-completed-reply formatter, and the live long-reply path — and
        the three copies had already drifted before this was extracted (one
        was missing the exception guard the others had). ``getSpecific``
        (underneath ``conf.get``) already treats a non-channel ``target``
        (e.g. a PM nick) as no channel scope, so callers do not need to
        pre-check with ``ircutils.isChannel`` themselves. ``network`` is
        optional because the pending-delivery poller has no ``irc`` handle
        to scope by; the live reply path does and passes it.

        Falls back to the unscoped default on any registry error, and to
        ``400`` when the registry value itself is unset — its default is
        ``0``, which several tests rely on this method to paper over.
        """
        try:
            return conf.get(conf.supybot.reply.mores.length, channel=target, network=network) or 400
        except Exception:
            # Swallowed on purpose (a scoped read must never break a reply),
            # but logged: otherwise a genuinely broken registry value looks
            # exactly like a bot that has never been configured.
            self.log.warning(
                "reply mores length: scoped read failed, falling back to the global value",
                exc_info=True,
            )
            return conf.supybot.reply.mores.length() or 400

    def _build_irc_lookup_tool(self, irc: callbacks.Irc, channel: str = ""):
        """Build the per-request ``irc_lookup`` tool schema + handler.

        Same shape as :meth:`_build_bridge_tool` and injected the same way:
        the handler closes over the live ``irc`` so it can send LIST / NAMES
        and wait on the numerics. ``channel`` is where the request came from:
        kind=history reads only that channel. Returns
        ``([schema], {"irc_lookup": fn})``.
        """
        from .assistant import ToolResult

        # kind=history ships channel chatter to the provider, so
        # privacyOptInOnly keeps it out of the schema entirely.
        history_on = bool(channel) and not self.registryValue("privacyOptInOnly", channel)
        schema = {
            "type": "function",
            "function": {
                "name": "irc_lookup",
                "description": (
                    "Look up live IRC network state. kind='channels' lists "
                    "public channels with user counts and topics (target is an "
                    "optional channel name or glob like '#linux*'; omit it to "
                    "see the busiest channels). kind='names' lists who is "
                    "currently in one channel (target required, must start "
                    "with #), including channels the bot has not joined. "
                    "kind='whois' looks up one user by nick (target required): "
                    "their host, real name, server, channels, account, idle "
                    "time and whether they are an IRC operator. Use it whenever "
                    "someone asks who a nick is, whether they are online, or "
                    "what channels they are in. kind='whowas' is for a nick "
                    "that is gone: when they left, their host and quit-time "
                    "server. kind='who' lists a channel's members with who is "
                    "away, an oper, an op/voice, a bot, and their account. "
                    "kind='topic' gives a channel's topic and who set it when. "
                    "kind='network' gives user/oper/channel counts, the server "
                    "software and the admin contact. kind='ctcp_version' asks a "
                    "nick's IRC client what it is; kind='ctcp_ping' measures "
                    "their lag (many clients ignore both)."
                    + (
                        " kind='history' returns the recent lines of THIS "
                        "channel from the server's history, newest last (count "
                        "up to 100, default 50): use it for 'what did I miss', "
                        "'summarize the last hour', or what someone said earlier."
                        if history_on
                        else ""
                    )
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "kind": {
                            "type": "string",
                            "enum": [
                                "channels",
                                "names",
                                "whois",
                                "whowas",
                                "who",
                                "topic",
                                "network",
                                "ctcp_version",
                                "ctcp_ping",
                                *(["history"] if history_on else []),
                            ],
                        },
                        "target": {
                            "type": "string",
                            "description": (
                                "Channel name or glob (channels), channel (names, "
                                "who, topic), or nick (whois, whowas, ctcp_*). "
                                + (
                                    "Omit for network and history."
                                    if history_on
                                    else "Omit for network."
                                )
                            ),
                        },
                        **(
                            {
                                "count": {
                                    "type": "integer",
                                    "description": "Lines of history to fetch (1-100).",
                                }
                            }
                            if history_on
                            else {}
                        ),
                    },
                    "required": ["kind"],
                },
            },
        }

        def handler(arguments: dict[str, Any]) -> ToolResult:
            kind = str(arguments.get("kind", "")).strip().lower()
            target = str(arguments.get("target", "") or "").strip()
            # One grep answers "did the model call the tool?" — the timing
            # line only says tool_calls=N, not which.
            self.log.info(
                "irc_lookup: kind=%s target=%r network=%s", kind, target, self._network_of(irc)
            )
            if kind == "whois":
                if not target or len(target) > 64 or not ircutils.isNick(target, strictRfc=False):
                    return ToolResult(content=json.dumps({"error": "target must be a nick"}))
                who = self._query_whois(irc, target)
                if who is None:
                    return ToolResult(content=json.dumps({"error": self._IRC_QUERY_SILENT}))
                if who.error is not None:
                    return ToolResult(content=json.dumps({"error": who.error}))
                envelope = {
                    "status": "ok",
                    "nick": who.nick,
                    "user": who.user,
                    "host": who.host,
                    "realname": ircquery.clean_text(who.realname, 120),
                    "server": who.server,
                    "server_info": ircquery.clean_text(who.server_info, 120),
                    "channels": who.channels[:50],
                    "account": who.account,
                    "oper": who.oper,
                    "away": ircquery.clean_text(who.away, 120) if who.away else None,
                    "idle_seconds": who.idle_seconds,
                    "signon": who.signon,
                }
                return ToolResult(content=json.dumps(envelope))
            if kind == "channels":
                rows = self._query_channels(irc)
                if rows is None:
                    err = self._irc_queries.last_error(self._network_of(irc), "LIST")
                    return ToolResult(content=json.dumps({"error": err or self._IRC_QUERY_SILENT}))
                if target:
                    rows = [r for r in rows if ircquery.match_channel(r.name, target)]
                rows.sort(key=lambda r: (-r.users, r.name.lower()))
                envelope = {
                    "status": "ok",
                    "total": len(rows),
                    "channels": [
                        {
                            "name": r.name,
                            "users": r.users,
                            "topic": ircquery.clean_text(r.topic, 120),
                        }
                        for r in rows[:25]
                    ],
                }
                return ToolResult(content=json.dumps(envelope))
            if kind == "names":
                if not ircutils.isChannel(target):
                    return ToolResult(
                        content=json.dumps({"error": "target must be a channel name like #chan"})
                    )
                result = self._query_names(irc, target)
                if result is None:
                    return ToolResult(content=json.dumps({"error": self._IRC_QUERY_SILENT}))
                if result.error is not None:
                    return ToolResult(content=json.dumps({"error": result.error}))
                envelope = {
                    "status": "ok",
                    "channel": target,
                    "count": len(result.nicks),
                    "nicks": result.nicks[:100],
                }
                return ToolResult(content=json.dumps(envelope))
            if kind == "history" and not history_on:
                return ToolResult(
                    content=json.dumps(
                        {"error": "history lookup is off: this channel is privacyOptInOnly"}
                    )
                )
            extra = self._irc_lookup_more(irc, kind, target, channel, arguments.get("count"))
            if extra is not None:
                return ToolResult(content=json.dumps(extra))
            return ToolResult(content=json.dumps({"error": "unknown kind"}))

        return [schema], {"irc_lookup": handler}

    def _irc_lookup_more(
        self, irc: callbacks.Irc, kind: str, target: str, channel: str, count: Any
    ) -> dict[str, Any] | None:
        """irc_lookup kinds past channels/names/whois; None for an unknown kind."""
        silent = {"error": self._IRC_QUERY_SILENT}
        needs_nick = kind in ("whowas", "ctcp_version", "ctcp_ping")
        if needs_nick and (
            not target or len(target) > 64 or not ircutils.isNick(target, strictRfc=False)
        ):
            return {"error": "target must be a nick"}
        if kind in ("who", "topic") and not ircutils.isChannel(target):
            return {"error": "target must be a channel name like #chan"}

        if kind == "whowas":
            got = self._query_collect(
                irc, "whowas", target, ircmsgs.IrcMsg(command="WHOWAS", args=(target, "3"))
            )
            if got is None:
                return silent
            rows, error = got
            if error:
                return {"error": error}
            entries: list[dict[str, Any]] = []
            for numeric, args in rows:
                if numeric == "314" and len(args) >= 5:
                    entries.append(
                        {
                            "nick": args[0],
                            "user": args[1],
                            "host": args[2],
                            "realname": ircquery.clean_text(args[4], 120),
                        }
                    )
                elif numeric == "312" and entries and len(args) >= 3:
                    entries[-1]["server"] = args[1]
                    entries[-1]["last_seen"] = args[2]
                elif numeric == "301" and entries and len(args) >= 2:
                    entries[-1]["away"] = ircquery.clean_text(args[1], 120)
            return {"status": "ok", "nick": target, "entries": entries}

        if kind == "who":
            got = self._query_collect(
                irc,
                "who",
                target,
                ircmsgs.IrcMsg(command="WHO", args=(target, f"%tcuhnfar,{_WHO_LOOKUP_TOKEN}")),
            )
            if got is None:
                return silent
            rows, error = got
            if error:
                return {"error": error}
            members = []
            for user, host, nick, flags, account, realname in (r for r in rows if len(r) == 6):
                members.append(
                    {
                        "nick": nick,
                        "away": flags.startswith("G"),
                        "oper": "*" in flags,
                        "status": "".join(c for c in flags if c in "@%+"),
                        "bot": "B" in flags,
                        "account": None if account == "0" else account,
                        "host": f"{user}@{host}",
                        "realname": ircquery.clean_text(realname, 60),
                    }
                )
            return {
                "status": "ok",
                "channel": target,
                "count": len(members),
                "members": members[:100],
            }

        if kind == "topic":
            if target in irc.state.channels:
                got = self._query_collect(
                    irc, "topic", target, ircmsgs.IrcMsg(command="TOPIC", args=(target,))
                )
                if got is None:
                    return silent
                rows, error = got
                if error:
                    return {"status": "ok", "channel": target, "topic": None, "note": error}
                out: dict[str, Any] = {"status": "ok", "channel": target}
                for numeric, args in rows:
                    if numeric == "332" and len(args) >= 2:
                        out["topic"] = ircquery.clean_text(args[1], 300)
                    elif numeric == "333" and len(args) >= 3:
                        out["set_by"] = args[1].split("!", 1)[0]
                        with contextlib.suppress(ValueError):
                            out["set_at"] = datetime.fromtimestamp(int(args[2]), UTC).isoformat()
                return out
            # Not joined: Limnoria's state would throw on the 332, and LIST
            # carries the topic (without the setter) for any public channel.
            listed = self._query_channels(irc)
            if listed is None:
                return silent
            for r in listed:
                if ircutils.strEqual(r.name, target):
                    return {
                        "status": "ok",
                        "channel": r.name,
                        "topic": ircquery.clean_text(r.topic, 300),
                        "note": "setter unknown: the bot is not in this channel",
                    }
            return {"error": "no such public channel"}

        if kind == "network":
            got = self._query_collect(
                irc,
                "network",
                "*",
                ircmsgs.IrcMsg(command="LUSERS"),
                ircmsgs.IrcMsg(command="VERSION"),
                ircmsgs.IrcMsg(command="ADMIN"),
                partial=True,
            )
            if got is None:
                return silent
            rows, _ = got
            lines: dict[str, list[str]] = {"users": [], "server": [], "admin": []}
            for numeric, args in rows:
                if numeric in ("251", "255", "265", "266") and args:
                    lines["users"].append(args[-1])
                elif numeric in ("252", "253", "254") and len(args) >= 2:
                    lines["users"].append(f"{args[0]} {args[1]}")
                elif numeric == "351" and len(args) >= 2:
                    lines["server"].append(f"{args[0]} on {args[1]}")
                elif numeric in ("257", "258", "259") and args:
                    lines["admin"].append(ircquery.clean_text(args[-1], 120))
            return {"status": "ok", "network": self._network_of(irc), **lines}

        if kind in ("ctcp_version", "ctcp_ping"):
            what = "VERSION" if kind == "ctcp_version" else "PING"
            stamp = str(int(time.time() * 1000))
            payload = f"\x01{what}\x01" if what == "VERSION" else f"\x01PING {stamp}\x01"
            got = self._query_collect(
                irc,
                f"ctcp-{what}",
                target,
                ircmsgs.privmsg(target, payload),
                timeout=self._CTCP_TIMEOUT,
            )
            if got is None or not got[0]:
                return {
                    "status": "ok",
                    "nick": target,
                    "reply": None,
                    "note": "no reply: the client ignores CTCP or the nick is offline",
                }
            body = str(got[0][0])
            if what == "VERSION":
                return {"status": "ok", "nick": target, "client": ircquery.clean_text(body, 200)}
            try:
                lag_ms = int(time.time() * 1000) - int(body.split()[0])
            except (ValueError, IndexError):
                return {"status": "ok", "nick": target, "reply": ircquery.clean_text(body, 60)}
            return {"status": "ok", "nick": target, "lag_ms": lag_ms}

        if kind == "history":
            if not ircutils.isChannel(channel):
                return {"error": "history works in a channel, not a private message"}
            if target and not ircutils.strEqual(target, channel):
                return {"error": f"history only covers this channel ({channel})"}
            try:
                n = max(1, min(100, int(count))) if count is not None else 50
            except (TypeError, ValueError):
                n = 50
            if "draft/chathistory" not in getattr(irc.state, "capabilities_ack", set()):
                return {"error": "the server has not granted chat history to the bot"}
            got = self._query_collect(
                irc,
                "history",
                channel,
                ircmsgs.IrcMsg(command="CHATHISTORY", args=("LATEST", channel, "*", str(n))),
            )
            if got is None:
                return silent
            rows, _ = got
            out_lines = []
            for ts, nick, text, _msgid in rows:
                hhmm = ts[11:16] if len(ts) >= 16 else "?"
                body = text
                if body.startswith("\x01ACTION ") and body.endswith("\x01"):
                    body = f"* {body[8:-1]}"
                out_lines.append(f"{hhmm} <{nick}> {ircquery.clean_text(body, 300)}")
            return {"status": "ok", "channel": channel, "times": "UTC", "lines": out_lines}

        return None

    @staticmethod
    def _valid_react_emoji(emoji: str) -> bool:
        """True for something that reads as one emoji, not as text.

        ASCII is refused outright: a model handing over ":thumbsup:" or
        "thumbs up" would otherwise go out as a literal-text reaction.
        """
        if not emoji or len(emoji) > _REACT_EMOJI_MAX_CHARS:
            return False
        return not any(ord(c) < 128 or c.isspace() for c in emoji)

    def _build_react_tool(self, irc: callbacks.Irc, msg: IrcMsg, caller: Caller):
        """Build the per-request ``react`` tool schema + handler.

        Same shape as :meth:`_build_irc_lookup_tool`. With no ``nick`` the
        reaction lands on the message the request replied to (``+draft/reply``)
        or, without one, on the request itself; with a nick, on that nick's
        latest line in this channel (see _remember_last_msgid).
        Returns ``([schema], {"react": fn})``.
        """
        from .assistant import ToolResult

        schema = {
            "type": "function",
            "function": {
                "name": "react",
                "description": (
                    "Add an emoji reaction to a message in this channel. With no "
                    "nick it reacts to the message the user replied to, or to "
                    "their own message when it is not a reply; with a nick, to "
                    "that person's most recent message here. A "
                    "successful reaction IS your whole reply: no text is sent "
                    "after it. Use one real emoji character, not a :name:."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "emoji": {"type": "string", "description": "One emoji, e.g. 👍"},
                        "nick": {
                            "type": "string",
                            "description": (
                                "Whose latest message to react to. Omit for the "
                                "message the user replied to, or their own."
                            ),
                        },
                    },
                    "required": ["emoji"],
                },
            },
        }

        def handler(arguments: dict[str, Any]) -> ToolResult:
            emoji = str(arguments.get("emoji", "") or "").strip()
            nick = str(arguments.get("nick", "") or "").strip()
            self.log.info("react: emoji=%r nick=%r network=%s", emoji, nick, self._network_of(irc))
            if not self._valid_react_emoji(emoji):
                return ToolResult(
                    content=json.dumps({"error": "emoji must be one emoji character, like 👍"})
                )
            where = caller.reply_target
            in_channel = not caller.is_private
            target = where
            tags = getattr(msg, "server_tags", None) or {}
            trigger_msgid = tags.get("msgid") or ""
            reply_msgid = tags.get("+draft/reply") or ""
            if not nick and reply_msgid:
                # A client reply points at the message they mean.
                target_label = f"the message {msg.nick} replied to"
                msgid = reply_msgid
            elif not nick or ircutils.strEqual(nick, msg.nick):
                # The requester's latest line is the one being answered.
                target_label = f"{msg.nick}'s message"
                msgid = trigger_msgid
            else:
                msgid = ""
                if in_channel:
                    key = (irc.network, ircutils.toLower(where), ircutils.toLower(nick))
                    with self._last_msgids_lock:
                        msgid = self._last_msgids.get(key, "")
                if msgid:
                    target_label = f"{nick}'s message"
                elif trigger_msgid and self.llm_service.send_reaction(
                    irc, target, trigger_msgid, "❌"
                ):
                    # Nothing of theirs to point at (they have not spoken
                    # since the bot started, or this is a PM): a ❌ on the
                    # request says so, and ends the turn like any reaction.
                    # Handing the model an error instead let grok write a
                    # fake "[reacted 🫡 to Larry's last message]" line.
                    return ToolResult(
                        content=json.dumps(
                            {
                                "status": "ok",
                                "message": (
                                    f"no recent message from {nick} to react to; "
                                    f"reacted ❌ to {msg.nick}'s request"
                                ),
                            }
                        )
                    )
                else:
                    return ToolResult(
                        content=json.dumps({"error": f"no recent message from {nick} here"})
                    )
            if not msgid or not self.llm_service.send_reaction(irc, target, msgid, emoji):
                return ToolResult(
                    content=json.dumps(
                        {"error": "reactions are unavailable here; answer in text instead"}
                    )
                )
            return ToolResult(
                content=json.dumps(
                    {"status": "ok", "message": f"reacted {emoji} to {target_label}"}
                )
            )

        return [schema], {"react": handler}

    def _build_redact_tool(self, irc: callbacks.Irc, msg: IrcMsg, caller: Caller):
        """Build the per-request ``delete_last_reply`` tool (IRCv3 REDACT).

        Takes back the bot's latest reply where the request came from: every
        line it sent within _REPLY_BURST_SECONDS of its newest one. Silent
        like react. Returns ``([schema], {"delete_last_reply": fn})``.
        """
        from .assistant import ToolResult

        schema = {
            "type": "function",
            "function": {
                "name": "delete_last_reply",
                "description": (
                    "Delete your own most recent reply in this channel (it "
                    "disappears for people whose clients support message "
                    "deletion). Use when someone asks you to delete, remove or "
                    "take back what you just said. A successful delete IS your "
                    "whole reply: no text is sent after it."
                ),
                "parameters": {"type": "object", "properties": {}},
            },
        }

        def handler(arguments: dict[str, Any]) -> ToolResult:
            target = caller.reply_target
            self.log.info("delete_last_reply: target=%r network=%s", target, self._network_of(irc))
            if "draft/message-redaction" not in getattr(irc.state, "capabilities_ack", set()):
                return ToolResult(
                    content=json.dumps({"error": "the server has not granted message deletion"})
                )
            key = (irc.network, ircutils.toLower(target))
            with self._own_msgids_lock:
                lines = getattr(self, "_own_lines", {}).get(key)
                if not lines:
                    burst: list[str] = []
                else:
                    newest = lines[-1][1]
                    burst = [m for m, ts in lines if newest - ts <= _REPLY_BURST_SECONDS]
                    # Forget them, so a second "delete that" reaches further back.
                    while lines and lines[-1][0] in burst:
                        lines.pop()
            if not burst:
                return ToolResult(
                    content=json.dumps(
                        {"error": "no reply of mine here to delete (I forget them on restart)"}
                    )
                )
            for msgid in burst:
                self._safe_queue(
                    irc,
                    ircmsgs.IrcMsg(command="REDACT", args=(target, msgid, f"asked by {msg.nick}")),
                )
            n = len(burst)
            return ToolResult(
                content=json.dumps(
                    {
                        "status": "ok",
                        "message": f"deleted my last reply ({n} line{'s' if n != 1 else ''})",
                    }
                )
            )

        return [schema], {"delete_last_reply": handler}

    def _build_notify_tool(self, irc: callbacks.Irc, msg: IrcMsg, caller: Caller):
        """Build the per-request ``notify_when_online`` tool (IRC MONITOR).

        Returns ``([schema], {"notify_when_online": fn})``.
        """
        from .assistant import ToolResult

        schema = {
            "type": "function",
            "function": {
                "name": "notify_when_online",
                "description": (
                    "Tell the asker when a nick comes online (the server "
                    "pushes the sign-on; nothing polls). action='watch' "
                    "(default) starts a watch and says whether they are "
                    "already on; action='cancel' stops one; action='list' "
                    "shows the asker's watches. A watch fires once, then ends; "
                    "it lapses after 7 days."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "nick": {"type": "string", "description": "Who to watch for."},
                        "action": {"type": "string", "enum": ["watch", "cancel", "list"]},
                    },
                },
            },
        }

        def handler(arguments: dict[str, Any]) -> ToolResult:
            action = str(arguments.get("action", "") or "watch").strip().lower()
            nick = str(arguments.get("nick", "") or "").strip()
            net = self._network_of(irc)
            requester = caller.identity.raw_nick
            target = caller.reply_target
            self.log.info("notify_when_online: action=%s nick=%r network=%s", action, nick, net)

            def result(payload: dict[str, Any]) -> ToolResult:
                return ToolResult(content=json.dumps(payload))

            if action == "list":
                mine = self.db.list_nick_watches(net, requester, max_age=_WATCH_MAX_AGE)
                return result({"status": "ok", "watching": mine})
            if not nick or len(nick) > 64 or not ircutils.isNick(nick, strictRfc=False):
                return result({"error": "nick must be an IRC nick"})
            if ircutils.strEqual(nick, irc.nick):
                return result({"error": "that is me"})
            if action == "cancel":
                if not self.db.delete_nick_watch(net, nick, requester):
                    return result({"error": f"you were not watching {nick}"})
                if nick.lower() not in self.db.list_nick_watches(net):
                    self._safe_queue(irc, ircmsgs.IrcMsg(command="MONITOR", args=("-", nick)))
                return result({"status": "ok", "message": f"stopped watching {nick}"})
            if action != "watch":
                return result({"error": "action must be watch, cancel or list"})
            mine = self.db.list_nick_watches(net, requester, max_age=_WATCH_MAX_AGE)
            if nick.lower() not in mine and len(mine) >= _WATCHES_PER_REQUESTER:
                return result({"error": f"you already watch {len(mine)} nicks; cancel one first"})
            got = self._query_collect(
                irc, "monitor", nick, ircmsgs.IrcMsg(command="MONITOR", args=("+", nick))
            )
            status = got[0][0] if got and got[0] else None
            if got and got[1]:
                return result({"error": got[1]})
            if status == "730":
                if nick.lower() not in self.db.list_nick_watches(net):
                    self._safe_queue(irc, ircmsgs.IrcMsg(command="MONITOR", args=("-", nick)))
                return result({"status": "ok", "online": True, "message": f"{nick} is online now"})
            # Offline (731), or the server stayed silent: watch either way.
            self.db.add_nick_watch(net, nick, requester, target)
            return result(
                {
                    "status": "ok",
                    "online": False,
                    "message": f"{nick} is offline; {requester} will be told here when they sign on",
                }
            )

        return [schema], {"notify_when_online": handler}

    # -- MONITOR numerics ----------------------------------------------------

    @staticmethod
    def _monitor_nicks(msg: IrcMsg) -> list[str]:
        """Nicks out of a 730/731 target list: ``nick!user@host,nick2,...``."""
        if len(msg.args) < 2:
            return []
        return [t.split("!", 1)[0] for t in msg.args[-1].split(",") if t]

    def do730(self, irc: callbacks.Irc, msg: IrcMsg) -> None:  # noqa: N802
        """RPL_MONONLINE: answer a pending watch, or deliver the sign-on."""
        net = self._network_of(irc)
        for nick in self._monitor_nicks(msg):
            if self._irc_queries.feed("monitor", net, nick, msg.command):
                self._irc_queries.close("monitor", net, nick)
                continue
            if msg.command != "730":
                continue
            try:
                watchers = self.db.pop_nick_watches(net, nick, max_age=_WATCH_MAX_AGE)
            except Exception:
                self.log.exception("notify_when_online: watch lookup failed for %s", nick)
                continue
            for requester, target in watchers:
                self.log.info(
                    "notify_when_online: %s is on, telling %s in %s", nick, requester, target
                )
                self._safe_queue(
                    irc, ircmsgs.privmsg(target, f"{requester}: {nick} is online now.")
                )
            if watchers:
                self._safe_queue(irc, ircmsgs.IrcMsg(command="MONITOR", args=("-", nick)))

    do731 = do730  # RPL_MONOFFLINE

    def do734(self, irc: callbacks.Irc, msg: IrcMsg) -> None:  # noqa: N802
        """ERR_MONLISTFULL: ``<me> <limit> <targets> :<text>``."""
        if len(msg.args) >= 3:
            for nick in msg.args[2].split(","):
                self._irc_queries.close(
                    "monitor", self._network_of(irc), nick, error="the bot's watch list is full"
                )

    def _resend_monitors(self, irc: callbacks.Irc) -> None:
        """MONITOR lists die with the connection: re-add every stored watch."""
        try:
            nicks = self.db.list_nick_watches(self._network_of(irc), max_age=_WATCH_MAX_AGE)
        except Exception:
            self.log.exception("notify_when_online: could not reload watches")
            return
        chunk: list[str] = []
        for nick in [*nicks, None]:
            if nick is not None and len(",".join([*chunk, nick])) < 400:
                chunk.append(nick)
                continue
            if chunk:
                self._safe_queue(
                    irc, ircmsgs.IrcMsg(command="MONITOR", args=("+", ",".join(chunk)))
                )
            chunk = [nick] if nick is not None else []

    def _build_bridge_tool(self, irc, msg, channel: str, trace: list | None = None):
        """Build the per-request Limnoria bridge tool schemas + handlers.

        Returns ``(None, None)`` when the bridge is disabled, the allowlist is
        empty, or no allowed command is currently exposable. Otherwise returns
        ``([run_schema, search_schema], {"run_limnoria_command": ...,
        "search_bridge_commands": ...})`` for injection into
        ``assistant_completion`` via ``extra_tools`` / ``extra_handlers``.

        When ``trace`` is provided, each successful or failed dispatch appends
        a ``(plugin, command, args, status)`` tuple — used by the optional
        ``bridgeDebugInChannel`` reply footer. ``search_bridge_commands`` calls
        append ``("bridge", "search", query, status)``.
        """
        if not self.registryValue("bridgeEnabled", channel):
            return None, None
        allowed = frozenset(self.registryValue("bridgeAllowedPlugins", channel) or [])
        if not allowed:
            # Empty registry value → fall back to the curated default set.
            # Limnoria persists every registered value to disk so a code
            # default change wouldn't reach existing operators on upgrade
            # (see DEFAULT_ALLOWED_PLUGINS docstring in limnoria_bridge.py).
            allowed = limnoria_bridge.DEFAULT_ALLOWED_PLUGINS
        allow_mutating = bool(self.registryValue("bridgeAllowMutating", channel))

        commands = list(
            limnoria_bridge.enumerate_commands(irc, msg, allowed, allow_mutating=allow_mutating)
        )
        if not commands:
            return None, None

        table = "\n".join(
            f"- {c.plugin}.{c.command}"
            + (f" — {c.arg_syntax}" if c.arg_syntax else "")
            + (f" — {c.description}" if c.description else "")
            for c in commands
        )
        # Footer: if the gate is closed AND any allowlisted plugin has at least
        # one mutating leaf, hint that more commands exist behind the gate.
        # Skips the hint for pure-read allowlists (Time, Math, etc.) where no
        # writes would be hidden.
        mutating_plugins = {p for (p, _leaf) in limnoria_bridge.MUTATING_COMMANDS}
        allowed_canonical = {p.lower() for p in allowed}
        hidden_writes_present = not allow_mutating and bool(allowed_canonical & mutating_plugins)
        footer = (
            "\n\n(write commands hidden — set bridgeAllowMutating True to expose)"
            if hidden_writes_present
            else ""
        )
        schema = {
            "type": "function",
            "function": {
                "name": "run_limnoria_command",
                "description": (
                    "Run a Limnoria plugin command on the user's behalf. "
                    "Available commands:\n" + table + footer
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "plugin": {
                            "type": "string",
                            "description": "Plugin name (e.g. Misc).",
                        },
                        "command": {
                            "type": "string",
                            "description": "Leaf command name (e.g. ping).",
                        },
                        "args": {
                            "type": "string",
                            "description": (
                                "Argument string passed to the plugin command. "
                                "Empty string for commands taking no arguments."
                            ),
                        },
                    },
                    "required": ["plugin", "command", "args"],
                },
            },
        }
        search_schema = {
            "type": "function",
            "function": {
                "name": "search_bridge_commands",
                "description": (
                    "Substring-search the available Limnoria bridge commands "
                    "by plugin name, command name, argument syntax, and "
                    "description text. Use this when Misc.apropos returns "
                    "nothing — apropos only matches command NAMES, this also "
                    "scans docstrings. Returns up to `limit` matches as "
                    "Plugin.command — argsyntax — description rows."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "Whitespace-separated keywords to match "
                                "against command name, syntax, and description."
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max matches to return (1-25).",
                            "minimum": 1,
                            "maximum": 25,
                        },
                    },
                    "required": ["query"],
                },
            },
        }

        from .assistant import ToolResult

        def handler(arguments):
            plugin_name = str(arguments.get("plugin", ""))
            command_name = str(arguments.get("command", ""))
            arg_string = str(arguments.get("args", ""))
            envelope = limnoria_bridge.dispatch(
                irc,
                msg,
                plugin=plugin_name,
                command=command_name,
                arg_string=arg_string,
                allowed=allowed,
                allow_mutating=allow_mutating,
            )
            if trace is not None:
                status = (
                    "ok" if envelope.get("status") == "ok" else f"err:{envelope.get('error', '?')}"
                )
                trace.append((plugin_name, command_name, arg_string, status))
            return ToolResult(content=json.dumps(envelope))

        def search_handler(arguments):
            query = str(arguments.get("query", "")).strip()
            try:
                raw_limit = int(arguments.get("limit", 10))
            except (TypeError, ValueError):
                raw_limit = 10
            limit = max(1, min(25, raw_limit))
            if not query:
                envelope = {"error": "query is required"}
            else:
                matches = limnoria_bridge.search_commands(commands, query, limit=limit)
                envelope = {
                    "status": "ok",
                    "matches": [
                        {
                            "plugin": c.plugin,
                            "command": c.command,
                            "arg_syntax": c.arg_syntax,
                            "description": c.description,
                        }
                        for c in matches
                    ],
                }
            if trace is not None:
                status = (
                    "ok" if envelope.get("status") == "ok" else f"err:{envelope.get('error', '?')}"
                )
                trace.append(("bridge", "search", query, status))
            return ToolResult(content=json.dumps(envelope))

        return [schema, search_schema], {
            "run_limnoria_command": handler,
            "search_bridge_commands": search_handler,
        }

    @staticmethod
    def _format_bridge_debug_footer(trace: list) -> str:
        """Render a one-line debug footer for the optional bridgeDebugInChannel mode.

        This footer is sent to a public channel. Bridge args are LLM-generated
        and may carry secrets (e.g. an API key in a URL), so the raw arg_string
        is NEVER echoed — only its length, as a leak-free "args were passed"
        signal. Operators who need the content can read it from the DEBUG logs.
        """
        if not trace:
            return ""
        parts = []
        for plugin_name, command_name, arg_string, status in trace:
            call = f"{plugin_name}.{command_name}"
            if arg_string:
                call += f" ({len(arg_string)} chars)"
            parts.append(f"{call} [{status}]")
        return "[bridge: " + " ; ".join(parts) + "]"

    @staticmethod
    def _trim_long_reply_teaser(teaser: str, max_chars: int) -> str:
        """Collapse and trim a teaser so the link reply stays on one line."""
        teaser = " ".join(teaser.split()) or _FULL_ANSWER_LABEL
        truncated = truncate_to_word_boundary(teaser, max_chars)
        if truncated != teaser:
            truncated = truncated.rstrip(" ,;:-")
        if truncated:
            return truncated
        # The label is a fallback, not an exemption from the budget.
        return _FULL_ANSWER_LABEL if max_chars <= 0 or len(_FULL_ANSWER_LABEL) <= max_chars else ""

    @staticmethod
    def _fallback_long_reply_teaser(text: str, max_chars: int) -> str:
        """Return a deterministic one-line teaser when LLM summarization fails."""
        teaser = next(
            (line.strip() for line in text.splitlines() if line.strip()), _FULL_ANSWER_LABEL
        )
        teaser = teaser.lstrip("#").strip()
        teaser = teaser.lstrip("-* ").strip()
        return LLM._trim_long_reply_teaser(teaser, max_chars)

    def _finish_irc_line(
        self,
        content: str,
        *,
        inline: str | None,
        allowed: int,
        teaser_fn: Callable[[str, int], str],
        save_fn: Callable[[str], str | None],
        nick_prefix: str = "",
        teaser_cap: int | None = None,
        save_failed_fallback: str | None = None,
    ) -> str:
        """Finish ``content`` as exactly one IRC wire-line string.

        The single inline-vs-pastebin policy shared by the live reply path
        (``_send_long_reply``) and the pending-recovery path
        (``_format_pending_completed_reply``): if ``inline`` fits one
        wire-line it is returned as-is; otherwise ``content`` is saved via
        ``save_fn`` and replaced with ``nick_prefix`` + teaser + URL suffix.
        Callers keep their own send mechanism and teaser strategy.

        Args:
            content: Full body to pastebin on overflow.
            inline: One-line inline candidate, or None when the content may
                never go inline (multi-line live replies pastebin
                unconditionally rather than collapsing).
            allowed: Wire-line budget (``supybot.reply.mores.length``).
            teaser_fn: ``(content, max_chars) -> str`` producing the final,
                already-trimmed teaser for the link line.
            save_fn: ``(content) -> url | None`` pastebin hook.
            nick_prefix: Optional ``"nick: "`` prefix, reserved out of the
                teaser budget and prepended to the link line.
            teaser_cap: Optional hard cap on teaser length
                (``longReplyTeaserMaxChars``).
            save_failed_fallback: Line to return when the save fails; None
                sends the teaser alone rather than risking an oversized body.
        """
        if inline is not None and len(ircutils.wrap(inline, allowed) or [inline]) <= 1:
            return inline
        url = save_fn(content)
        if not url and save_failed_fallback is not None:
            return save_failed_fallback
        suffix = f" - {_FULL_ANSWER_LABEL}: {url}" if url else ""
        label_line = f"{nick_prefix}{_FULL_ANSWER_LABEL}: {url}"
        # Bytes, because ``allowed`` is what ircutils.wrap measures above: a
        # character budget let a CJK or emoji teaser run the line to three
        # times the budget and the server cut the URL off the end. Whatever
        # teaser_fn returns — an LLM summary included — is clipped to it.
        # Proved for any teaser_fn in docs/formal/IrcLine.lean.
        max_bytes = max(0, allowed - len(suffix.encode()) - len(nick_prefix.encode()))
        max_chars = max_bytes if teaser_cap is None else min(teaser_cap, max_bytes)
        if max_chars <= 0 and url:
            return label_line
        raw = teaser_fn(content, max_chars)
        teaser = truncate_to_byte_budget(raw, max_bytes)
        if teaser != raw:
            teaser = teaser.rstrip(" ,;:-")
        if not teaser and url:
            return label_line
        return f"{nick_prefix}{teaser}{suffix}"

    def _format_pending_completed_reply(
        self, nick: str, content: str, target: str | None = None
    ) -> str:
        """One-line delivery text for a completed ask/fallback pending result.

        Shares ``_finish_irc_line`` with ``_send_long_reply``. Pending
        delivery runs on the poll thread with no ``irc``/``msg`` and is already
        a deferred-recovery fallback, so a deterministic (non-LLM) teaser is
        used. Falls back to the inline/collapsed line when the save fails, so we
        never emit a raw multi-line body (the outer ``_collapse_for_irc`` still
        guards flood); the only residual risk is server-side truncation when the
        HTTP save itself is unavailable.
        """
        inline = f"{nick}: {content}"
        collapsed = self._collapse_for_irc(inline) or inline
        # Channel-scoped like _send_long_reply (no irc handle here, so no
        # network scope) — the global read gave channels with a custom mores
        # length inconsistent inline-vs-pastebin decisions on recovery.
        allowed = self._reply_mores_length(target)
        return self._finish_irc_line(
            content,
            inline=collapsed,
            allowed=allowed,
            teaser_fn=self._fallback_long_reply_teaser,
            save_fn=self.llm_service.save_markdown_to_http,
            nick_prefix=f"{nick}: ",
            save_failed_fallback=collapsed,
        )

    def _safe_queue(self, irc: callbacks.Irc, msg: IrcMsg) -> bool:
        """Thread-safe wrapper around irc.queueMsg for worker-thread sends.

        Limnoria's IrcMsgQueue.enqueue (irclib.py:245) mutates internal
        state without an explicit lock and the dedup check is TOCTOU.
        With the new executor pool, multiple workers may call queueMsg
        concurrently — serialize them on the plugin side and short-circuit
        cleanly when the plugin is shutting down.

        Returns True if the message was queued, False if dropped due
        to the plugin closing. Callers that mutate durable state on
        successful send must check the return.
        """
        if self._llm_executor.closing:
            self.log.debug(
                "safe_queue dropped (closing) command=%s",
                getattr(msg, "command", "?"),
            )
            return False
        with self._irc_send_lock:
            irc.queueMsg(msg)
        return True

    @staticmethod
    def _safe_privmsg(target: str, text: str, reply_to: str | None = None) -> IrcMsg:
        """Build a PRIVMSG whose body is neutralized against IRC injection.

        ``reply_to`` is the msgid of the message being answered. Supplying it
        attaches the IRCv3 ``+draft/reply`` tag, which is what makes a client
        thread the line under the original request instead of dropping it into
        the channel unattached. The chat path gets this free from
        ``irc.reply``; a deferred delivery has no ``msg`` to reply to, so the
        msgid has to be carried from submission and re-attached here. Callers
        must check ``message-tags`` is ACKed before passing it — a tag the
        server never negotiated has no business on the wire.

        Routes the body through Limnoria's ``ircutils.safeArgument`` (which
        repr()s any string containing CR, LF, or NUL) so model- or
        user-derived text cannot smuggle a second IRC command onto the wire.
        This is the raw-queue counterpart to the ``safeArgument`` that
        ``irc.reply`` applies on the chat-loop path: ``_safe_queue`` callers
        construct ``ircmsgs.privmsg`` directly and would otherwise rely solely
        on an ``IrcMsg.__init__`` assertion that disappears under ``python -O``.
        Callers should still ``_collapse_for_irc`` multi-line bodies first so a
        legitimate answer is split into a readable line rather than repr()'d.
        """
        out = ircmsgs.privmsg(target, ircutils.safeArgument(text))
        if reply_to:
            out = ircmsgs.IrcMsg(msg=out, server_tags={"+draft/reply": reply_to})
        return out

    def _safe_reply(
        self,
        irc: callbacks.Irc,
        text: str,
        *,
        prefixNick: bool = False,  # noqa: N803  (mirrors irc.reply kwarg)
    ) -> bool:
        """Thread-safe wrapper around ``irc.reply`` for worker-thread sends.

        ``irc.reply`` ultimately reaches the same ``IrcMsgQueue.enqueue`` as
        ``irc.queueMsg``, so it must serialize on the same ``_irc_send_lock``:
        ``ask``/``code``/``draw`` release Limnoria's command RLock via
        ``_allow_concurrent`` and reply from concurrent worker threads.

        Returns True if the reply was sent, False if dropped due to the
        plugin closing.
        """
        if self._llm_executor.closing:
            self.log.debug("safe_reply dropped (closing)")
            return False
        with self._irc_send_lock:
            irc.reply(text, prefixNick=prefixNick)
        return True

    def _safe_error(
        self,
        irc: callbacks.Irc,
        text: str,
        *,
        prefixNick: bool = False,  # noqa: N803  (mirrors irc.error kwarg)
        Raise: bool = False,  # noqa: N803  (mirrors irc.error kwarg)
    ) -> None:
        """Thread-safe wrapper around ``irc.error`` for worker-thread sends.

        ``irc.error`` ultimately reaches the same ``IrcMsgQueue.enqueue`` as
        ``irc.queueMsg``, so it must serialize on the same ``_irc_send_lock``:
        ``ask``/``code``/``draw`` release Limnoria's command RLock via
        ``_allow_concurrent`` and error from concurrent worker threads.

        The ``Raise`` kwarg preserves Limnoria flow-control semantics: if the
        caller passes ``Raise=True`` the exception is re-raised after the lock
        is released so Limnoria's command dispatcher still sees it.
        """
        if self._llm_executor.closing:
            self.log.debug("safe_error dropped (closing)")
            return
        # Only NestedCommandsIrcProxy.error knows Raise; the ReplyIrcProxy an
        # addressed (non-command) turn carries forwards every kwarg to
        # _makeReply, which raises TypeError on it. Prod 2026-10-10: an empty
        # Haiku reply crashed here and the user saw nothing at all.
        extra = {"Raise": True} if Raise else {}
        with self._irc_send_lock:
            irc.error(text, prefixNick=prefixNick, **extra)

    def _send_long_reply(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        text: str,
        *,
        prefixNick: bool = False,  # noqa: N803  (mirrors irc.reply kwarg)
        style: str = "answer",
    ) -> bool:
        """Reply with ``text`` inline when it is short, pastebinning the rest.

        Returns False when the line or batch could not be queued.

        One wire-line goes out as-is. A reply that wraps to at most
        ``multilineMaxLines`` wire-lines goes out as a single draft/multiline
        batch when the cap is negotiated — a click-through for two lines of
        answer was the complaint. Anything longer is saved to the bot's HTTP
        server and the channel receives a one-line teaser plus the URL. When
        the save fails, we collapse to a single line ("teaser only") so we
        never send a raw multi-line body.
        """
        target = msg.channel if msg.channel else msg.nick
        allowed = self._reply_mores_length(target, network=irc.network)

        logical_lines = [line for line in text.split("\n") if line.strip()]
        # (wire text, is a wrap-continuation of the previous logical line)
        chunks = [
            (piece, i > 0)
            for line in logical_lines
            for i, piece in enumerate(ircutils.wrap(line, allowed) or [line])
        ]
        max_lines = int(self.registryValue("multilineMaxLines", target) or 0)
        if 1 < len(chunks) <= max_lines and self._multiline_negotiated(irc):
            return self._send_multiline_batch(irc, msg, chunks, prefixNick=prefixNick)

        # Without a batch, multi-line answers pastebin (never collapsed into
        # one line); only a sole non-blank logical line may go inline, and
        # _finish_irc_line still byte-wraps it against the wire budget.
        inline = logical_lines[0] if len(logical_lines) == 1 else None

        # One summary serves double duty: the page <title> (echoed by URL-title
        # bots in-channel) and the inline IRC teaser. Computed lazily so the
        # inline path never pays for the LLM call, and memoized so the title
        # costs nothing extra over the teaser we already make.
        configured_max_chars = int(self.registryValue("longReplyTeaserMaxChars", target) or 220)
        summary_cache: list[str | None] = [None]

        def _summary() -> str:
            if summary_cache[0] is None:
                summary_cache[0] = self.llm_service.summarize_for_irc(
                    text, channel=target, max_chars=configured_max_chars
                )
            return summary_cache[0] or ""

        line = self._finish_irc_line(
            text,
            inline=inline,
            allowed=allowed,
            teaser_fn=lambda t, mc: self._trim_long_reply_teaser(
                _summary() or self._fallback_long_reply_teaser(t, mc), mc
            ),
            save_fn=lambda t: self.llm_service.save_markdown_to_http(
                t, title=_summary(), style=style
            ),
            teaser_cap=configured_max_chars,
        )
        return self._safe_reply(irc, line, prefixNick=prefixNick)

    @staticmethod
    def _multiline_negotiated(irc: callbacks.Irc) -> bool:
        """True when a draft/multiline batch may be sent on this connection."""
        return bool(
            conf.supybot.protocols.irc.experimentalExtensions()
            and "draft/multiline" in irc.state.capabilities_ack
        )

    def _send_multiline_batch(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        chunks: list[tuple[str, bool]],
        *,
        prefixNick: bool = False,  # noqa: N803  (mirrors irc.reply kwarg)
    ) -> bool:
        """Send ``chunks`` as one draft/multiline batch addressed like irc.reply.

        Each line is built by Limnoria's own ``_makeReply`` so the target,
        PRIVMSG-vs-NOTICE choice (``withNoticeWhenPrivate``) and nick prefix
        match a plain ``irc.reply``. Limnoria's ``queueMultilineBatches`` can't
        be used: it takes one batch-wide ``concat`` flag, and the tag is only
        valid on wrap-continuations — distinct lines must stay distinct.

        The spec forbids client tags inside the batch, so the first line's
        tags (``+draft/reply``) move to the opening BATCH. Callers keep the
        batch far under the server's ``max-bytes``, so it is never split.
        """
        lines = [
            callbacks._makeReply(irc, msg, text, prefixNick=prefixNick and i == 0)
            for i, (text, _concat) in enumerate(chunks)
        ]
        name = ircutils.makeLabel()
        batch = [
            ircmsgs.IrcMsg(
                command="BATCH",
                args=("+" + name, "draft/multiline", lines[0].args[0]),
                server_tags=dict(lines[0].server_tags),
            )
        ]
        for i, (line, (_text, concat)) in enumerate(zip(lines, chunks, strict=True)):
            line.server_tags = {"batch": name}
            if concat and i > 0:
                line.server_tags["draft/multiline-concat"] = None
            batch.append(line)
        batch.append(ircmsgs.IrcMsg(command="BATCH", args=("-" + name,)))

        if self._llm_executor.closing:
            self.log.debug("multiline batch dropped (closing)")
            return False
        with self._irc_send_lock:
            irc.queueBatch(batch)
        return True

    def _record_last_verse_line(self, irc: callbacks.Irc, channel: str, text: str, result) -> None:
        """Remember the bot's last VERSE line per (network, channel) for reaction
        attribution. No-op for non-verse turns. Never disturbs the reply path."""
        if not getattr(result, "was_verse", False):
            return
        try:
            with self._irc_send_lock:
                self._last_bot_line[(irc.network, ircutils.toLower(channel))] = {
                    "text": text,
                    "ts": time.time(),
                }
        except Exception:
            self.log.exception("last_bot_line store failed")

    def _dispatch_assistant_reply(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        result: AssistantResult,
        *,
        caller: Caller,
        response: str,
        suppress_reminder_mutations: bool = False,
    ) -> tuple[str, bool]:
        """Send the reply for an assistant result.

        Returns ``(response_for_context, delivered)``. When the response is
        empty (and not suppressed by a successful reminder mutation), the
        helper emits ``irc.error`` and returns ``(response, False)``; the
        caller passes ``delivered`` to ``_store_context_and_log_usage``, which
        books the spend either way and stores history only for a delivery.

        Reminder-mutation suppression (``ask`` only) is checked BEFORE the
        empty-response error branch; that is the existing behaviour and must
        be preserved.
        """
        nick, channel = caller.identity.key, caller.reply_target
        if result.model and caller.channel:
            served = getattr(self, "_served_models", None)
            if served is None:
                served = self._served_models = {}
            served[(self._network_of(irc), caller.channel.lower())] = (
                result.model,
                time.time(),
            )
        if suppress_reminder_mutations and result.last_successful_tool in _REMINDER_MUTATION_TOOLS:
            # Any text is dropped, not just empty text: told it may stay
            # quiet, gemini narrated the silence instead (2026-09-25:
            # "An empty response is submitted to stay quiet...").
            self.log.info(
                "suppressing post-reminder-mutation reply tool=%s chars=%s %s/%s",
                result.last_successful_tool,
                len(result.final_text_after_tools.strip()),
                channel,
                nick,
            )
            # Nothing goes to IRC — the emoji reaction is the ack — but the
            # stored turn records that the tool ran, so a follow-up message
            # does not look like the original request went unanswered. Carry
            # the tool's own words too: a bare "[reminder set]" says nothing
            # about what was set or when, and a follow-up that revises the
            # time ("wait no make it 30s") then has nothing in this user's
            # history to attach to and binds to whatever else in the channel
            # mentioned a duration (observed 2026-09-03: it answered about
            # another user's request).
            label = _REMINDER_MUTATION_CONTEXT_NOTES[result.last_successful_tool]
            detail = result.last_tool_message.strip()
            return (f"[{label}: {detail}]" if detail else f"[{label}]"), True

        # verse_storybook delivers its illustrated page link from a background
        # job, so the assistant's post-tool reply is intentionally empty (see
        # the short-circuit in assistant_completion). Suppress it silently —
        # no message, no empty-response error; the async link is the reply.
        if (
            result.last_successful_tool == "verse_storybook"
            and not (result.final_text_after_tools or "").strip()
        ):
            self.log.info("suppressing verse_storybook interim reply %s/%s", channel, nick)
            return response, True

        # A react-only turn (see the short-circuit in assistant_completion):
        # the emoji is the reply. Nothing goes to IRC, but the stored turn
        # says what happened, for the same reason the reminder note does.
        if result.last_successful_tool in SILENT_TOOLS and not (response or "").strip():
            detail = result.last_tool_message.strip()
            self.log.info("react-only reply %s/%s: %s", channel, nick, detail)
            return (f"[{detail}]" if detail else "[reacted]"), True

        # Text shaped like that stored note is the model copying it, not a
        # reaction: a real one short-circuits above with no text. 2026-10-10
        # #afternet: grok's react missed and it posted "[reacted 🫡 to Larry's
        # last message]". Send the ❌ it should have got, and store nothing so
        # the fake line does not become the next exemplar.
        if _FAKE_REACT_NOTE_RE.match(response or ""):
            self.log.warning("fake react note dropped %s/%s: %r", channel, nick, response[:120])
            self._react(irc, msg, "❌")
            return response, False

        if not response or not response.strip():
            self._safe_error(irc, _("The model returned an empty response. Please try again."))
            return response, False

        action_text = self._extract_action(irc, response)
        if action_text:
            action_text = self._prefix_reply_icons(action_text, result)
            self.log.info("sending action to %s/%s", channel, nick)
            target = caller.reply_target
            # safeArgument for the same reason as _safe_privmsg: this is a
            # raw-queue send of model-derived text, and IrcMsg's own CR/LF/NUL
            # assertion vanishes under python -O. sanitize_output already
            # removed the CTCP delimiter that safeArgument does NOT cover.
            if not self._safe_queue(
                irc, ircmsgs.action(target, ircutils.safeArgument(action_text))
            ):
                return response, False
            self._record_last_verse_line(irc, channel, action_text, result)
            return f"* {irc.nick} {action_text}", True

        display_response = self._prefix_reply_icons(response, result)
        self.log.info("replying to %s/%s", channel, nick)
        # Verse scenes that overflow to a paste render with the storybook
        # parchment theme — they're stories, not Q&A answers.
        if not self._send_long_reply(
            irc,
            msg,
            display_response,
            prefixNick=False,
            style="story" if getattr(result, "was_verse", False) else "answer",
        ):
            return response, False
        self._record_last_verse_line(irc, channel, display_response, result)
        return response, True

    def _reply_and_book(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        result: AssistantResult,
        *,
        caller: Caller,
        command: str,
        text: str,
        response: str,
        suppress_reminder_mutations: bool = False,
    ) -> None:
        """Send an assistant turn's reply, then book it.

        The turn is billed before anything is sent, so a send that raises
        still books its row (as undelivered) before the exception goes on.
        """
        try:
            response, delivered = self._dispatch_assistant_reply(
                irc,
                msg,
                result,
                caller=caller,
                response=response,
                suppress_reminder_mutations=suppress_reminder_mutations,
            )
        except Exception:
            try:
                self._store_context_and_log_usage(
                    caller,
                    command,
                    text,
                    response,
                    result,
                    irc,
                    delivered=False,
                )
            except Exception:
                self.log.exception("%s usage log failed after a failed reply", command)
            raise
        self._store_context_and_log_usage(
            caller,
            command,
            text,
            response,
            result,
            irc,
            delivered=delivered,
        )

    def _build_request_context(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        preflight: PreflightResult,
        *,
        entry_route: str,
        profile: str,
    ) -> AssistantRequestContext:
        """Normalize route metadata into a shared assistant request context."""
        capabilities = frozenset(
            capability
            for capability in _REQUEST_CONTEXT_CAPABILITIES
            if ircdb.checkCapability(msg.prefix, capability)
        )
        return AssistantRequestContext(
            entry_route=entry_route,
            profile=profile,
            caller=preflight.caller,
            is_owner="owner" in capabilities,
            capabilities=capabilities,
        )

    def _verse_triggered(
        self,
        channel: str,
        store: VerseStore,
        avatar_id: int,
        message_text: str,
    ) -> bool:
        """True when ``message_text`` should drop into verse mode.

        Two signals, either sufficient:
        1. The message names a known active in-universe entity (character,
           place, or item) other than the speaker's own avatar. Reuses
           ``store.match_entities_in_text`` (whole-word, alias-aware,
           capitalized-stoplist rules).
        2. The channel's ``verseTriggerRegex`` (case-insensitive ``re.search``)
           matches. Empty regex disables this signal; a malformed regex is
           ignored rather than raised into the message path.
        """
        if any(e.id != avatar_id for e in store.match_entities_in_text(message_text)):
            return True
        pattern = self.registryValue("verseTriggerRegex", channel)
        if pattern:
            try:
                if re.search(pattern, message_text, re.IGNORECASE):
                    return True
            except re.error:
                pass
        return False

    @staticmethod
    def _find_caller_avatar(store, account: str | None, nick: str) -> int | None:
        """Resolve a caller's avatar entity id: account link wins, then nick."""
        return (
            store.find_avatar_by_account(account) if account else None
        ) or store.find_avatar_by_nick(nick)

    def _verse_route_for(
        self,
        channel: str,
        nick: str,
        account: str | None,
        message_text: str,
        prefix: str | None = None,
        *,
        force_roleplay: bool = False,
    ) -> VerseRoute | None:
        """Return a VerseRoute when the message should enter verse ROLEPLAY mode,
        or None to fall through to the normal chat path.

        Gates (applied in order):
        1. verseEnabled must be True for the channel.
        2. User must hold the llm.verse capability.
        3. OOC messages (wrapped in ((...)) or with a leading //) bypass the
           verse engine.
        4. The caller must have an avatar in the verse.
        5. Roleplay is entered EXPLICITLY (``force_roleplay=True``: the @rp
           command, a live sticky @rp session, or the one-shot ambient-prose
           promotion) — a bare keyword/entity mention no longer arms roleplay
           mode on its own. Mentions instead inject canon *facts* into the chat
           turn (see ``build_verse_context_block`` in
           ``_dispatch_with_verse_routing``). This separates the canon layer
           (read/write, cheap) from the roleplay persona (explicit, heavy).
        """
        if not ircutils.isChannel(channel) or not self.registryValue("verseEnabled", channel):
            return None
        # Prefer the real msg.prefix: hostmask-authed accounts can resolve to
        # a different user than a synthetic nick!*@* pattern would. The
        # synthetic fallback remains for direct calls (test harness).
        hostmask = prefix or f"{nick}!*@*"
        if not ircdb.checkCapability(hostmask, "llm.verse"):
            return None  # capability fallthrough — quiet
        if is_ooc(message_text):
            return None
        # Avatar lookup: account takes priority, then nick.
        store = self._get_or_create_verse_store(channel)
        avatar_id = self._find_caller_avatar(store, account, nick)
        if avatar_id is None:
            return None  # User opted into the channel but isn't in the verse → chat path.
        # Explicit-only: roleplay mode is entered by the @rp command, not by
        # a bare mention. Without the explicit signal, fall through to chat
        # (where the mention still injects canon facts).
        if not force_roleplay:
            return None
        # Personal data: only an authenticated caller has a persona.
        persona = (self.db.get_avatar_persona(account) or "") if account else ""
        system_prompt = build_verse_system_prompt(
            store,
            avatar_id,
            persona,
            roster_max_chars=self.registryValue("verseRosterMaxChars", channel),
            message_text=message_text,
            style_exemplars=self.registryValue("verseStyleExemplars", channel),
        )
        max_actors = self.registryValue("verseAutoEntityMaxNamesPerCall", channel)
        tools = make_verse_tool_specs(
            max_actors=max_actors,
            storybook=bool(self.registryValue("verseStorybookEnabled", channel)),
        )
        return VerseRoute(avatar_id, system_prompt, tools, store)

    def _verse_context_for(
        self, preflight: PreflightResult, text: str, *, for_chat: bool = False
    ) -> str | None:
        """Facts-only canon block to inject into a NORMAL chat turn, or None.

        The retrieval side of the canon layer: when a verse-channel message
        references canon (an entity mention or the channel's trigger keyword),
        return the compact lore block (``build_verse_context_block``) so an
        ordinary chat answer is grounded in canon — WITHOUT entering roleplay
        mode. Returns None when the channel has no verse or nothing is
        referenced. Best-effort: a failure here never breaks the chat turn.

        ``for_chat`` appends a flavour nudge so a grounded ANSWER reads in the
        world's voice; left False for the ``@draw`` grounding path, where an
        "answer in voice" instruction has no business in an image prompt.
        """
        channel = preflight.caller.channel
        if channel is None or not self.registryValue("verseEnabled", channel):
            return None
        try:
            store = self._get_or_create_verse_store(channel)
            avatar_id = self._find_caller_avatar(
                store, preflight.caller.identity.account, preflight.caller.identity.raw_nick
            )
            # Retrieval trigger: only inject when the message actually references
            # canon. -1 stands in for a non-avatar speaker (no self to exclude),
            # so lore grounding works for opted-out users too.
            if not self._verse_triggered(
                channel, store, avatar_id if avatar_id is not None else -1, text
            ):
                return None
            block = build_verse_context_block(
                store,
                text,
                avatar_id=avatar_id,
                roster_max_chars=self.registryValue("verseRosterMaxChars", channel),
            )
            if not block:
                return None
            # Slice 2 nudge: when chat-path canon recording is enabled for an
            # opted-in avatar, invite the model to save a genuinely new durable
            # fact (sparingly) via verse_record. Only when the handler is
            # actually wired (see _verse_chat_record_handler) — same gate.
            if avatar_id is not None and self.registryValue("verseChatRecordEnabled", channel):
                block = f"{block}\n{_VERSE_CHAT_RECORD_NUDGE}"
            if for_chat:
                block = f"{block}\n{_VERSE_CHAT_FLAVOUR_NUDGE}"
            return block
        except Exception:
            self.log.exception("verse context injection failed (non-fatal) channel=%s", channel)
            return None

    def _verse_grounded_overlay(self, profile: str, channel: str, verse_context: str) -> str:
        """Lore block layered onto a profile's channel overlay, for the overlay slot.

        ``system_prompt=`` on an assistant_completion is the personality-overlay
        slot; the profile's framework prompt and tools are still added
        downstream by assistant_completion. So a media planner that wants its
        request grounded in canon puts the facts-only block here, on top of
        whatever overlay the channel configures for that profile.

        The media profiles currently set ``overlay_setting=None`` — they build
        their system prompt without reading a channel key — and ``None`` is not
        a registry name: passing it to registryValue raises TypeError inside
        Limnoria's own ``registry.split``. Hence the guard rather than a
        straight lookup, so this keeps working whichever way a profile is
        configured.
        """
        overlay_key = PROFILES[profile].overlay_setting
        overlay = self.registryValue(overlay_key, channel) if overlay_key else ""
        return "\n\n".join(p for p in [overlay, verse_context] if p)

    def _subject_dossier_for(self, nick: str, channel: str, text: str) -> str | None:
        """Appearance facts for the real subjects ``text`` names, or None.

        The research half of the media planners' grounding, and the sibling of
        ``_verse_context_for``: canon covers what the channel invented, this
        covers what the world already contains. Best-effort in the same way — a
        failure here costs the request its detail, never its render, so every
        exit that is not "facts to inject" returns None and says nothing to the
        channel.

        Books its own usage row. The pre-stage runs on a different model from
        the planner it feeds (gemini and grok respectively, in this
        deployment), so folding its spend into the planner's row would file one
        provider's tokens under the other's name. A call that cost nothing gets
        no row: an exception is already in the log and a zero row only
        pollutes the spend table.

        Caller's contract: this makes a blocking LLM call, so wrap it in
        ``with self._allow_concurrent(), self._llm_executor.permit():``.
        """
        from .prompts import SUBJECT_DOSSIER_BLOCK_FOOTER, SUBJECT_DOSSIER_BLOCK_HEADER

        if not self.registryValue("subjectResearchEnabled", channel):
            return None
        try:
            dossier = self.llm_service.subject_dossier(text, channel=channel)
        except Exception:
            self.log.exception("subject research failed (non-fatal) channel=%s", channel)
            return None
        if dossier.prompt_tokens or dossier.completion_tokens or dossier.cost:
            self.db.log_usage(
                nick,
                channel,
                "dossier",
                dossier.model,
                dossier.prompt_tokens,
                dossier.completion_tokens,
                dossier.cost,
                prompt=text,
            )
        if not dossier.text:
            return None
        return "\n".join([SUBJECT_DOSSIER_BLOCK_HEADER, dossier.text, SUBJECT_DOSSIER_BLOCK_FOOTER])

    @staticmethod
    def _ambient_verse_intent(text: str) -> str:
        """Classify an ambient verse-mention by requested OUTPUT.

        A canon mention is ALWAYS a story cue — the trigger word exists to get a
        tale, so we never try to guess that some phrasing "only" wants a short
        answer. There are exactly two exceptions, both of which are the user
        explicitly asking for pictures:
        - "illustrate" — an explicit request for an illustrated tale → the full
                         multi-image storybook, told from the same canon.
        - "draw"       — an explicit single-picture request → canon-grounded
                         image, no tale.
        - "story"      — EVERYTHING else, including questions ("who is X",
                         "what have the lads done today") → a full prose tale.

        Illustrate is checked first so "draw the story with pictures" opens the
        storybook rather than being read as a single-picture request.
        """
        t = (text or "").strip()
        if _ILLUSTRATE_INTENT_RE.search(t):
            return "illustrate"
        if _DRAW_INTENT_RE.search(t):
            return "draw"
        return "story"

    @staticmethod
    def _looks_like_question(text: str) -> bool:
        """True when the message reads as a question (interrogative opener or a
        trailing '?'). Used to route a real-world question to a straight,
        truthful answer instead of the channel's tall-tale voice — see the
        chat-path use of ``_FACTUAL_CHAT_OVERLAY``.
        """
        t = (text or "").strip()
        if not t:
            return False
        return bool(_QUESTION_SHAPE_RE.match(t)) or t.endswith("?")

    def _ambient_storybook_brief(
        self, msg: IrcMsg, preflight: PreflightResult, text: str
    ) -> str | None:
        """Brief to fire an illustrated storybook for an ambient mention that
        EXPLICITLY asks to be illustrated, or None to fall through.

        Returns ``text`` only when ALL hold: storybook enabled, the message is
        an explicit "illustrate"/comic/storybook request ("illustrate") — every
        other canon mention is a prose-first inline tale (see
        ``_ambient_inline_story``), NOT an image page — it references canon
        (``_verse_triggered``), and the @story-style spend gates pass (account +
        llm.draw + not on cooldown). Any miss → None → the caller uses the chat
        path (draw drawn), still verse-grounded. The cooldown check reserves the
        slot, matching the verse_storybook tool handler.
        """
        channel = preflight.caller.channel
        if channel is None or not self.registryValue("verseStorybookEnabled", channel):
            return None
        if self._ambient_verse_intent(text) != "illustrate":
            return None
        try:
            store = self._get_or_create_verse_store(channel)
            avatar_id = self._find_caller_avatar(
                store, preflight.caller.identity.account, preflight.caller.identity.raw_nick
            )
            if not self._verse_triggered(
                channel, store, avatar_id if avatar_id is not None else -1, text
            ):
                return None
        except Exception:
            self.log.exception("ambient storybook trigger check failed channel=%s", channel)
            return None
        # Spend gates, mirroring _storybook_handler: authenticated + llm.draw +
        # per-account cooldown (reserve-at-start). A miss falls through to chat.
        account = preflight.caller.identity.account
        if not account:
            return None
        if not ircdb.checkCapability(msg.prefix, "llm.draw"):
            return None
        cooldown = int(self.registryValue("verseStorybookCooldownSeconds", channel) or 0)
        if self._storybook_cooldown_active(account, cooldown):
            return None
        return text

    def _ambient_inline_story(self, preflight: PreflightResult, text: str) -> bool:
        """True when an ambient mention should become an INLINE prose tale.

        The prose-first ambient story, and the DEFAULT for any canon mention: a
        message that references canon, from an avatar-holder, is answered as a
        multi-paragraph tale posted straight in the channel (the verse
        completion) — no image, no pastebin page. Questions included: we do not
        try to guess that some phrasing wants a short answer instead. The only
        exceptions are the two explicit picture asks — "illustrate" (opens the
        image storybook) and "draw" (single canon-grounded picture) — which keep
        their own paths.

        The caller uses this to promote the turn to a one-shot verse-prose route
        (``force_roleplay`` for this call only — it does NOT arm sticky @rp).
        Requires an avatar because the prose is told in that avatar's voice; a
        non-avatar speaker falls through to grounded chat instead. Best-effort:
        any failure returns False (→ the caller's normal chat/draw paths stand).
        """
        channel = preflight.caller.channel
        if channel is None or not self.registryValue("verseEnabled", channel):
            return False
        if self._ambient_verse_intent(text) != "story":
            return False
        try:
            store = self._get_or_create_verse_store(channel)
            avatar_id = self._find_caller_avatar(
                store, preflight.caller.identity.account, preflight.caller.identity.raw_nick
            )
            if avatar_id is None:
                return False
            return self._verse_triggered(channel, store, avatar_id, text)
        except Exception:
            self.log.exception("ambient inline-story check failed channel=%s", channel)
            return False

    def _verse_chat_record_handler(
        self, preflight: PreflightResult
    ) -> Callable[[dict], _VerseToolResult] | None:
        """Live ``verse_record`` handler for the CHAT path, or None.

        Slice 2 of the canon layer: when ``verseChatRecordEnabled`` is on and the
        caller is an opted-in avatar, canon may accrue from ordinary
        (non-roleplay) chat — the model can call verse_record to save a durable
        fact, bound to the caller's avatar. Off by default, so chat-path
        verse_record stays denied and canon only grows during @rp roleplay.
        Best-effort: returns None on any failure (→ the denial handler stands).
        """
        channel = preflight.caller.channel
        if channel is None or not self.registryValue("verseEnabled", channel):
            return None
        if not self.registryValue("verseChatRecordEnabled", channel):
            return None
        try:
            store = self._get_or_create_verse_store(channel)
            avatar_id = self._find_caller_avatar(
                store, preflight.caller.identity.account, preflight.caller.identity.raw_nick
            )
            if avatar_id is None:
                return None
            max_actors = self.registryValue("verseAutoEntityMaxNamesPerCall", channel)
            handlers = make_verse_extra_handlers(store, avatar_id, max_actors=max_actors)
            return handlers.get("verse_record")
        except Exception:
            self.log.exception(
                "verse chat-record handler build failed (non-fatal) channel=%s", channel
            )
            return None

    # --- Slice 3: sticky @rp roleplay sessions -----------------------------

    def _roleplay_sticky_set(self, preflight: PreflightResult, on: bool) -> None:
        """Turn sticky @rp roleplay on/off for this caller+channel.

        On: store an expiry ``verseRoleplayStickyTtlSeconds`` in the future (inf
        when the TTL is 0 → never auto-expire). Off: drop the entry.
        """
        # Storage key: stable across nick changes once authenticated.
        channel = preflight.caller.reply_target
        key = (channel, preflight.caller.identity.key)
        with self._roleplay_sticky_lock:
            if not on:
                self._roleplay_sticky.pop(key, None)
                return
            ttl = int(self.registryValue("verseRoleplayStickyTtlSeconds", channel) or 0)
            self._roleplay_sticky[key] = (time.time() + ttl) if ttl > 0 else float("inf")

    def _roleplay_sticky_active(self, preflight: PreflightResult) -> bool:
        """True when sticky roleplay is live for this caller+channel.

        Sliding window: an active check refreshes the expiry, so ambient
        roleplay persists while you're talking and lapses only after
        ``verseRoleplayStickyTtlSeconds`` of silence. Expired entries are
        evicted lazily here.
        """
        # Storage key: stable across nick changes once authenticated.
        channel = preflight.caller.reply_target
        key = (channel, preflight.caller.identity.key)
        now = time.time()
        with self._roleplay_sticky_lock:
            exp = self._roleplay_sticky.get(key)
            if exp is None:
                return False
            if exp <= now:  # lapsed
                self._roleplay_sticky.pop(key, None)
                return False
            ttl = int(self.registryValue("verseRoleplayStickyTtlSeconds", channel) or 0)
            self._roleplay_sticky[key] = (now + ttl) if ttl > 0 else float("inf")
            return True

    def _build_verse_handlers_for_route(self, channel: str, route: VerseRoute) -> dict:
        """Build verse handlers for an existing ``VerseRoute``.

        Reads ``verseAutoEntityMaxNamesPerCall`` (channel-scoped) and
        threads it into ``make_verse_extra_handlers`` so the dispatch
        closure carries the per-channel cap.
        """
        max_actors = self.registryValue("verseAutoEntityMaxNamesPerCall", channel)
        return make_verse_extra_handlers(route.store, route.avatar_id, max_actors=max_actors)

    def _storybook_cooldown_active(self, account: str | None, cooldown: int) -> bool:
        """Single-slot per-account storybook cooldown (reserve-at-start).

        Shared by the verse_storybook tool and the @story command via the
        ``verse_storybook:<account>`` rate-bucket key, so both paths draw from
        ONE cooldown window and can't be stacked to double image spend. Reuses
        ``_rate_buckets``/``_rate_buckets_lock`` (the machinery behind
        ``_is_rate_limited``). Records the hit immediately when not limited; no
        account or non-positive cooldown disables limiting.
        """
        if cooldown <= 0 or not account:
            return False
        key = f"verse_storybook:{account}"
        now = time.time()
        with self._rate_buckets_lock:
            bucket = self._rate_buckets.get(key)
            if bucket and (now - bucket[-1]) < cooldown:
                return True
            if bucket is None:
                bucket = collections.deque(maxlen=1)
                self._rate_buckets[key] = bucket
            bucket.append(now)
            return False

    def _submit_storybook_job(
        self,
        *,
        caller: Caller,
        persona: str,
        brief: str,
        max_images: int | None = None,
    ) -> None:
        """Fire a background job that renders a storybook for ``brief`` and
        posts the resulting link to the caller's reply target (the channel, or
        their IRC nick in a PM). Fire-and-return — never raises
        to the caller. Shared by the verse_storybook tool and the @story command.

        Delivery looks up the live IRC connection via ``world.ircs`` (not a
        captured, possibly-stale ``irc``), mirroring ``_deliver_pending_result``.
        """
        # Record a canon event so an illustrated turn isn't invisible to verse
        # retention (the tool/@story paths short-circuit before any verse record
        # step). Only for verse channels — @story works anywhere, and a bare
        # @story in a non-verse channel or PM must not lazily create a verse DB.
        # Best-effort: never break delivery if the store/avatar lookup fails.
        # Also captures the canon roster so the story generator (which runs
        # outside the verse completion) uses established characters by name
        # instead of inventing them.
        channel = caller.reply_target
        nick, account = caller.identity.raw_nick, caller.identity.account
        world_context = ""
        try:
            if caller.channel and self.registryValue("verseEnabled", channel):
                store = self._get_or_create_verse_store(channel)
                world_context = build_story_world_context(
                    store, roster_max_chars=self.registryValue("verseRosterMaxChars", channel)
                )
                avatar_id = self._find_caller_avatar(store, account, nick)
                if avatar_id is not None:
                    store.record_user_event(
                        actor_id=avatar_id,
                        summary=(brief.strip()[:200] or "told an illustrated tale"),
                        actor_names=[],
                    )
        except Exception:
            self.log.exception("storybook canon-record failed (non-fatal) channel=%s", channel)

        # Recent channel scene so a thin brief can draw on what actually
        # happened. Channel history only (shared, who-said-what) — the caller's
        # personal history isn't relevant to a shared tale. Best-effort + capped;
        # never block delivery on it. Empty when context is off for the channel.
        scene_context = ""
        try:
            _personal, channel_history = self._gather_history(caller)
            if channel_history:
                scene_context = self.llm_service._format_channel_history(channel_history)[:2000]
        except Exception:
            self.log.exception(
                "storybook scene-context gather failed (non-fatal) channel=%s", channel
            )

        def _deliver(text: str) -> None:
            collapsed = self._collapse_for_irc(text) or text
            if not caller.is_private:
                for irc_conn in world.ircs:
                    if channel in irc_conn.state.channels:
                        self._safe_queue(irc_conn, self._safe_privmsg(channel, collapsed))
                        return
                return
            # PM: the reply target is the requester's nick, never in
            # state.channels — deliver on the first available connection.
            for irc_conn in world.ircs:
                if self._safe_queue(irc_conn, self._safe_privmsg(nick, collapsed)):
                    return

        def _job(b: str) -> None:
            try:
                res = self.llm_service.generate_storybook(
                    b,
                    channel=channel,
                    persona=persona,
                    world_context=world_context,
                    scene_context=scene_context,
                    max_images=max_images,
                )
                if res is not None:
                    # @story's command wrapper returns before generation, so the
                    # shared _store_context_and_log_usage path never sees the
                    # spend — log the (image-dominated) usage here instead.
                    try:
                        self.db.log_usage(
                            nick=caller.identity.key,
                            channel=channel,
                            command="story",
                            model=res.model,
                            prompt_tokens=res.prompt_tokens,
                            completion_tokens=res.completion_tokens,
                            cost=res.cost,
                            prompt=b[:200],
                        )
                    except Exception:
                        self.log.exception("storybook usage logging failed nick=%s", nick)
                    _deliver(f"the tale is told — {res.title}: {res.url}")
                else:
                    _deliver("the tale slipped away before it could be illustrated.")
            except Exception:
                self.log.exception("storybook job failed channel=%s nick=%s", channel, nick)

        try:
            self._llm_executor.submit("verse_storybook", _job, brief)
        except RecursiveSubmitError:
            _job(brief)

    def _storybook_handler(
        self,
        *,
        irc: callbacks.Irc,
        msg: IrcMsg,
        caller: Caller,
        persona: str,
    ) -> Callable[[dict], _VerseToolResult]:
        """Build the ``verse_storybook`` tool handler for one completion.

        Returns a synchronous ``Callable[[dict], _VerseToolResult]`` matching
        the verse handler contract. The returned closure captures the caller
        context plus a mutable per-turn counter (scoped to this one handler
        instance → one completion).

        Gating: this factory is only invoked when ``verseStorybookEnabled``
        is True for the channel (see ``_ask_impl`` merge site), so when the
        flag is off the handler is never built and the tool is never even
        advertised. The flag-off verse path is byte-identical to before.

        On invocation the handler enforces, in order: per-turn cap → account
        gate → ``llm.draw`` capability → per-account cooldown. On success it
        fires a background job (which generates the storybook and posts the
        URL to the channel) and immediately returns a "generating" ack so the
        verse model can acknowledge in character without blocking the loop.
        """
        # Per-turn counter, captured by reference via a single-element list so
        # the inner closure can mutate it. Scoped to this handler instance.
        channel = caller.reply_target
        account = caller.identity.account
        turn_count = [0]
        max_per_turn = int(self.registryValue("verseStorybookMaxPerTurn", channel) or 1)
        cooldown = int(self.registryValue("verseStorybookCooldownSeconds", channel) or 0)

        def _err(message: str) -> _VerseToolResult:
            return _VerseToolResult(content=json.dumps({"status": "error", "error": message}))

        def _call(args: dict) -> _VerseToolResult:
            # 1. Per-turn cap.
            if turn_count[0] >= max_per_turn:
                return _err("already told a tale this turn")
            # 2. Account gate.
            if not account:
                return _err("you must be authenticated to weave a storybook")
            # 3. Capability gate (image spend).
            if not ircdb.checkCapability(msg.prefix, "llm.draw"):
                return _err("you lack the standing to summon illustrations")
            # 4. Per-account cooldown (reserve-at-start).
            if self._storybook_cooldown_active(account, cooldown):
                return _err("the muse needs a moment")

            # TODO(storybook): daily image cap (verseStorybookDailyImageCap) —
            # no straightforward per-account daily image-count query exists on
            # the usage table; cooldown + per-turn cap already bound spend.

            # Reserve the per-turn slot only once we're actually firing.
            turn_count[0] += 1
            brief = ""
            if isinstance(args, dict):
                raw = args.get("brief")
                if isinstance(raw, str):
                    brief = raw
            self._submit_storybook_job(caller=caller, persona=persona, brief=brief)
            return _VerseToolResult(
                content=json.dumps(
                    {
                        "status": "ok",
                        "note": (
                            "The illustrated page is rendering in the background and "
                            "its link will be posted automatically the moment it is "
                            "ready. Do NOT announce that a story is coming, do NOT "
                            "mention drawing/rendering/waiting, and do NOT promise a "
                            "link. Just stay in character — reply with a brief beat or "
                            "say nothing further."
                        ),
                    }
                )
            )

        _call.__name__ = "_verse_handler_verse_storybook"
        return _call

    def _verse_edit_handler(
        self,
        *,
        msg: IrcMsg,
        store: VerseStore,
        account: str | None,
    ) -> Callable[[dict], _VerseToolResult]:
        """Build the ``verse_edit`` tool handler for one completion.

        The handler is bound to the TRIGGERING user's IRC prefix (``msg.prefix``)
        so the ``llm.verse.edit`` capability check reflects who actually drove
        this verse turn — NOT the avatar that happens to be active in the
        channel. This is the security-load-bearing gate: an unauthorized caller
        gets a refusal and nothing is written to the store.

        ``dispatch_verse_edit`` (the pure handler) enforces the
        constructive-ops whitelist and payload validation; this wrapper only
        resolves authorization and adapts the result dict to the
        ``_VerseToolResult`` JSON the assistant loop expects. A refused/error
        result is re-emitted with an ``error`` key so the loop's success
        detector (service.py: "error" not in parsed) counts it as a failure.
        """
        authorized = ircdb.checkCapability(msg.prefix, "llm.verse.edit")

        def _call(args: dict) -> _VerseToolResult:
            op = args.get("op") if isinstance(args, dict) else None
            payload = args.get("payload") if isinstance(args, dict) else None
            if not isinstance(payload, dict):
                payload = {}
            result = dispatch_verse_edit(
                store,
                op=op,
                payload=payload,
                authorized=authorized,
                account=account or "anon",
            )
            if result.get("status") != "ok":
                return _VerseToolResult(
                    content=json.dumps(
                        {
                            "status": result.get("status", "error"),
                            "error": result.get("detail", "verse_edit failed"),
                        }
                    )
                )
            return _VerseToolResult(content=json.dumps(result))

        _call.__name__ = "_verse_handler_verse_edit"
        return _call

    def _draw_for_assistant(
        self, irc: callbacks.Irc, msg: IrcMsg, prompt: str, *, aspect: str | None = None
    ) -> ToolCallbackResult:
        """Generate an image for the generate_image tool.

        The documented rule is that leaf tool handlers do not log usage —
        ``_store_context_and_log_usage`` writes one row for the turn. This is
        the exception, and the reason is that a usage row records exactly ONE
        model. Every other leaf spends on a text model close enough to the
        caller's that folding the cost in loses nothing worth having; this one
        spends on the image model, and folding it in files image spend under
        whatever chat model happened to answer.

        That is not hypothetical. Since 2026-04-11, when @draw was converted to
        run through assistant_completion like everything else, EVERY path to an
        image has gone through this callback — so from that date no row in the
        usage table attributed image spend correctly, and `GROUP BY model` read
        as if the bot had stopped generating images. Storybook already solved it
        the same way, logging its own row from the background job.

        So: one row per turn from the wrapper for the text, one row per image
        from here, each naming the model that actually spent.
        """
        from .assistant import ToolCallbackResult as _ToolCallbackResult

        result = self.llm_service.image_generation(prompt, irc=irc, msg=msg, aspect=aspect)
        self._log_image_usage(msg, prompt, result)
        return _ToolCallbackResult(
            not bool(result.error),
            result.content,
            reworded=result.rewritten_prompt is not None,
        )

    def _log_image_usage(self, msg: IrcMsg, prompt: str, result: ImageResult) -> None:
        """Write one usage row per provider call, under the image model.

        Usually that is one row. It is more when the safety-rewrite loop ran:
        each refusal it recovered from was a real, usually billed, call to the
        provider, and folding those into the delivered row hides both the
        refusal and the prompt that caused it. ``blocked_attempts`` carries them
        out, deliberately holding back the refusal that IS the returned result
        (a draw that ran out of rewrites) so that call is not booked twice.

        The bill is split, not duplicated: ``result.cost`` already includes
        every attempt, so the blocked rows take their share and the final row
        takes the remainder — which is where the rewriter's own text spend
        stays, under the image model, since the row names one model and the
        image is the one worth naming.

        Skips silently when nothing was spent and nothing was refused — a prompt
        rejected by ``validate_prompt`` or a missing API key never reaches the
        provider, and a zero row would only dilute the averages. A refusal that
        the provider did not charge for is not that: it reached the provider, so
        it gets its row. Never raises: an accounting write must not be able to
        sink a picture the user is waiting for.
        """
        blocked = result.blocked_attempts
        final_cost = max(0.0, result.cost - sum(attempt.cost for attempt in blocked))
        if not (blocked or result.cost or result.prompt_tokens or result.completion_tokens):
            return
        try:
            if result.error is None:
                status = "success"
            elif self._is_content_blocked_error(result.error):
                status = "content_blocked"
            else:
                status = "error"
            where = Caller.from_msg(msg)
            nick, channel = where.identity.raw_nick, where.reply_target
            for attempt in blocked:
                self.db.log_usage(
                    nick,
                    channel,
                    _IMAGE_USAGE_COMMAND,
                    result.model,
                    0,
                    0,
                    attempt.cost,
                    prompt=attempt.prompt[:200],
                    status="content_blocked",
                    error_detail=attempt.reason[:200],
                )
            self.db.log_usage(
                nick,
                channel,
                _IMAGE_USAGE_COMMAND,
                result.model,
                result.prompt_tokens,
                result.completion_tokens,
                final_cost,
                prompt=prompt[:200],
                status=status,
                error_detail=(result.error or "")[:200],
            )
        except Exception:
            self.log.exception("image usage logging failed")

    @staticmethod
    def _prefix_reply_icons(text: str, result: AssistantResult) -> str:
        """Prefix the reply with an icon per signal the turn raised.

        Both can apply at once — a grounded answer that also drew a reworded
        image — so they chain rather than choosing. Order is fixed so the same
        turn always renders the same way.
        """
        icons = []
        if getattr(result, "grounding_used", False):
            icons.append(GROUNDING_ICON)
        if getattr(result, "image_reworded", False):
            icons.append(REWORDED_ICON)
        return f"{' '.join(icons)} {text}" if icons else text

    def _resolve_caller(self, irc: callbacks.Irc, msg: IrcMsg) -> Caller:
        """Resolve the sender of ``msg`` to a :class:`Caller`.

        The one resolver: the account comes from the IRCv3 account-tag (or
        the session cache) via :meth:`_account_from_msg`, the nick and the
        channel from ``msg`` via :meth:`Caller.from_msg`. Nothing is moved
        between storage keys: rows written under a nick stay there, so an
        account can't claim what someone else wrote under its name.
        """
        return Caller.from_msg(msg, self._account_from_msg(irc, msg))

    def _require_caller(self, irc: callbacks.Irc, msg: IrcMsg) -> Caller | None:
        """Resolve the Caller, refusing one that isn't authenticated.

        Commands that read or write personal data need a personal key. When
        the caller has none, sends an error reply and returns None; callers
        should ``return`` immediately.
        """
        caller = self._resolve_caller(irc, msg)
        if caller.identity.personal_key is None:
            self._safe_error(irc, _("You must be identified to use this command."))
            return None
        return caller

    def _run_preflight(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        text: str,
        command: str,
        *,
        require_account: bool,
    ) -> PreflightResult:
        """Shared preflight check for all commands.

        Runs the following sequence:
        1. Caller resolution; an authenticated account is required or
           optional depending on the command.
        2. Tier resolution (owner/admin exempt, then trusted/registered/unregistered).
        3. Per-command, per-tier rate-limit check.

        When any check fails the method sends the appropriate IRC error,
        logs usage with the blocked status, and returns ``blocked=True``.

        Args:
            irc: IRC connection.
            msg: IRC message.
            text: User's prompt text (for usage logging).
            command: Command name (ask, code, draw).
            require_account: If True, an authenticated account is mandatory.

        Returns:
            PreflightResult carrying the resolved Caller, with blocked=False
            if the command should proceed.
        """
        caller = self._resolve_caller(irc, msg)
        if require_account and caller.identity.personal_key is None:
            self._safe_error(irc, _("You must be identified to use this command."))
            self.db.log_usage(
                caller.identity.raw_nick,
                caller.reply_target,
                command,
                "",
                0,
                0,
                0.0,
                prompt=text,
                status="auth_failure",
            )
            return PreflightResult(blocked=True, caller=caller)

        # --- tier-based rate limit check ---
        tier = self._resolve_tier(irc, msg)
        # Owner and admin are always exempt from rate limits
        if tier not in ("owner", "admin"):
            key = caller.identity.key
            if self._check_rate_limit(irc, command, key, key, caller.reply_target, text, tier=tier):
                return PreflightResult(blocked=True, caller=caller)

        return PreflightResult(blocked=False, caller=caller)

    def _is_rate_limited(self, command: str, account: str, now: float, *, tier: str) -> bool:
        """Check if a user exceeds the per-command rate limit.

        Evicts timestamps outside the configured window before checking.

        Args:
            command: Command name (ask, code, or draw).
            account: NickServ account name or nick-based identity.
            now: Current time (seconds since epoch).
            tier: User tier (trusted, registered, unregistered).

        Returns:
            True if the user has exceeded the rate limit.
        """
        max_count, window = self._get_tier_limits(command, tier)
        key = f"{command}:{account}"

        # count=0 means rate limiting is disabled for this tier. Drop any bucket
        # left over from a non-zero setting (or from a caller that recorded
        # before the limit was turned off) — nothing below this point ever
        # prunes it while the tier is disabled, so it would grow unbounded.
        if max_count == 0:
            with self._rate_buckets_lock:
                self._rate_buckets.pop(key, None)
            return False

        cutoff = now - window

        with self._rate_buckets_lock:
            bucket = self._rate_buckets.get(key)
            if bucket is None:
                return False

            # Evict expired entries
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()

            # Clean up idle keys so bucket map cannot grow forever.
            if not bucket:
                self._rate_buckets.pop(key, None)
                return False

            return len(bucket) >= max_count

    def _record_rate_limit_hit(self, command: str, account: str, now: float) -> None:
        """Record a request timestamp in the rate-limit bucket.

        Args:
            command: Command name.
            account: NickServ account name.
            now: Current time.
        """
        key = f"{command}:{account}"
        with self._rate_buckets_lock:
            bucket = self._rate_buckets.get(key)
            if bucket is None:
                bucket = collections.deque()
                self._rate_buckets[key] = bucket
            bucket.append(now)

    def _check_rate_limit(
        self,
        irc: callbacks.Irc | None,
        command: str,
        account: str,
        nick: str,
        channel: str,
        text: str,
        *,
        tier: str,
        silent: bool = False,
        record: bool = True,
        now: float | None = None,
    ) -> bool:
        """Check rate limit; optionally suppress user-facing error and usage row.

        When ``record=False`` the check only PEEKS at the bucket: scheduled
        fires (reminder actions, scheduled tasks) are skipped when the user
        has already maxed their own interactive use, but the fire itself must
        not consume interactive slots — a busy watch chain was rate-limiting
        its owner's live @ask.

        When ``silent=True``:
          - ``irc.error(...)`` is NOT called on overage.
          - ``db.log_usage(..., status="rate_limited")`` is NOT written.
          - ``irc`` may be None (action-fire path has no caller IRC connection).
          - ``nick``/``channel``/``text`` are still accepted but unused in the
            silent branch — kept in the signature for caller-site uniformity.

        ``now`` defaults to ``time.time()`` when not supplied — keeps the
        original non-silent signature working without forcing every caller
        to thread a timestamp.

        Args:
            irc: IRC connection (may be None when ``silent=True``).
            command: Command name.
            account: NickServ account name or nick-based identity.
            nick: Resolved identity for logging.
            channel: Channel name.
            text: Prompt text for logging.
            tier: User tier (trusted, registered, unregistered).
            silent: When True, suppress ``irc.error`` and ``db.log_usage``.
            now: Optional pre-computed timestamp; defaults to ``time.time()``.

        Returns:
            True if the request should be blocked.
        """
        if now is None:
            now = time.time()
        over_limit = self._is_rate_limited(command, account, now, tier=tier)

        # Record the hit so the window tracks correctly (unless peeking).
        # Skipped when the tier's limit is disabled (count 0 — e.g. the default
        # codeTrustedRateLimitCount): there is no window to track, and the entry
        # would sit in _rate_buckets unpruned for the life of the process.
        if record and self._get_tier_limits(command, tier)[0] > 0:
            self._record_rate_limit_hit(command, account, now)

        if not over_limit:
            return False

        enforce = self.registryValue("enforceRateLimits")
        max_count, window = self._get_tier_limits(command, tier)
        key = f"{command}:{account}"
        with self._rate_buckets_lock:
            count = len(self._rate_buckets.get(key, ()))

        if enforce:
            self.log.info(
                "rate_limited command=%s account=%s tier=%s count=%i limit=%i window=%ss",
                command,
                account,
                tier,
                count,
                max_count,
                window,
            )
            if not silent and irc is not None:
                self._safe_error(
                    irc, _("Rate limit exceeded for %s. Try again in %ds.") % (command, window)
                )
                self.db.log_usage(
                    nick,
                    channel,
                    command,
                    "",
                    0,
                    0,
                    0.0,
                    prompt=text,
                    status="rate_limited",
                )
            return True

        self.log.info(
            "rate_limit_shadow command=%s account=%s tier=%s count=%i limit=%i window=%ss",
            command,
            account,
            tier,
            count,
            max_count,
            window,
        )
        return False

    @staticmethod
    def _is_content_blocked_error(error: str | None) -> bool:
        """Return True if an error string indicates a content safety block.

        Checks for common keywords that LLM providers use when content
        is rejected for safety/moderation reasons.

        Args:
            error: Error message string from the LLM service, or None.

        Returns:
            True if the error looks like a content safety block.
        """
        if not error:
            return False
        lower = error.lower()
        # Phrase-level matching: a bare "content" substring also matches
        # "content-length mismatch" / "unsupported content type" and skewed
        # the usage-status analytics toward content_blocked.
        phrases = (
            "content policy",
            "content blocked",
            "content_blocked",
            "content filter",
            "content management policy",
            "moderation",
            "safety",
            "blocked",
        )
        return any(p in lower for p in phrases)

    @staticmethod
    def _month_start_ts() -> float:
        """Return the UNIX timestamp for midnight UTC on the 1st of the current month."""
        return (
            datetime.now(UTC).replace(day=1, hour=0, minute=0, second=0, microsecond=0).timestamp()
        )

    # Tier config key prefixes: tier -> config infix
    _TIER_CONFIG_PREFIX = {
        "trusted": "Trusted",
        "unregistered": "Unreg",
        "registered": "",  # base config (no prefix)
    }

    def _resolve_tier(self, irc: callbacks.Irc, msg: IrcMsg) -> str:
        """Classify a user into a rate-limit tier based on Limnoria capabilities.

        Checks capabilities from most to least privileged.

        Args:
            irc: IRC connection (for account lookup).
            msg: IRC message (uses msg.prefix for capability check).

        Returns:
            One of: "owner", "admin", "trusted", "registered", "unregistered".
        """
        prefix = msg.prefix
        if ircdb.checkCapability(prefix, "owner"):
            return "owner"
        if ircdb.checkCapability(prefix, "admin"):
            return "admin"
        if ircdb.checkCapability(prefix, "trusted"):
            return "trusted"
        account = self._account_from_msg(irc, msg)
        return "registered" if account else "unregistered"

    def _get_tier_limits(self, command: str, tier: str) -> tuple[int, int]:
        """Look up rate limit count and window for a command+tier.

        Args:
            command: Command name (ask, code, draw).
            tier: User tier (trusted, registered, unregistered).

        Returns:
            (max_count, window_seconds). max_count=0 means disabled.
        """
        infix = self._TIER_CONFIG_PREFIX.get(tier, "")
        count_key = f"{command}{infix}RateLimitCount"
        window_key = f"{command}{infix}RateLimitWindow"
        return self.registryValue(count_key), self.registryValue(window_key)

    def _is_old_message(self, msg: IrcMsg) -> bool:
        """Check if message predates bot startup (ZNC playback).

        Args:
            msg: IRC message

        Returns:
            True if message is older than bot startup time
        """
        if msg.time == 0:
            return False  # No timestamp = live message (not ZNC playback)
        return msg.time < self.startup_time

    def _get_context_enabled(self, channel: str) -> bool:
        """Check if context is enabled for a channel.

        Args:
            channel: Channel name

        Returns:
            True if context is enabled for this channel
        """
        return self.registryValue("contextEnabled", channel)

    def _gather_history(
        self,
        caller: Caller,
        *,
        max_age_seconds: int | None = None,
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        """Return (personal_history, channel_history) for ``caller``.

        Personal history is the caller's own thread, keyed on their storage
        key at their reply target. Channel history is the channel's shared
        buffer, which a private message does not have: a PM caller always
        gets ``[]``, so no one's PM can surface in another request, even
        under a nick the earlier sender has since given up.

        Returns ([], []) when context is disabled for the channel or when
        ``max_age_seconds`` is 0 (callers use 0 to opt out of context while
        keeping a positive freshness window configurable). A positive value
        filters stale conversations at the context layer.
        """
        nick, channel = caller.identity.key, caller.reply_target
        if not self._get_context_enabled(channel) or max_age_seconds == 0:
            return [], []
        ctx_cfg = self._get_context_config(channel)
        history = self.context.get_messages(
            nick, channel, config=ctx_cfg, max_age_seconds=max_age_seconds
        )
        if caller.channel is None:
            return history, []
        # The shared buffer labels lines with the nick people see, not the key.
        channel_history = self.context.get_channel_messages(
            caller.channel,
            exclude_nick=caller.identity.raw_nick,
            config=ctx_cfg,
            max_age_seconds=max_age_seconds,
        )
        return history, channel_history

    def _get_user_memories(self, nick: str) -> list[str]:
        """Get memory facts for a user as a list of strings."""
        if self.db is None:
            return []
        rows = self.db.get_memories(nick)
        return [row.fact for row in rows]

    def _schedule_memory_extraction(
        self, nick: str, channel: str, user_text: str, assistant_response: str
    ) -> None:
        """Schedule background memory extraction for a user interaction.

        Two-stage flow:

        1. The extractor proposes new candidate facts and/or reinforces
           existing ones. New entries land in ``memory_candidates`` with
           ``mentions=1``; reinforcements bump that counter.
        2. Once a candidate's ``mentions`` reaches
           ``memoryPromotionThreshold`` it is promoted to ``memories`` and
           removed from the candidate table. Untouched candidates older
           than ``memoryCandidateTTLDays`` are pruned on each pass.

        Args:
            nick: User's resolved identity
            channel: Channel where the interaction happened
            user_text: What the user said
            assistant_response: What the bot replied
        """
        try:
            if not self.registryValue("memoryEnabled", channel):
                return

            existing_rows = self.db.get_memories(nick)
            existing_facts = [r.fact for r in existing_rows]
            max_memories = self.registryValue("memoryMaxPerUser")

            if len(existing_rows) >= max_memories:
                return

            existing_candidates = self.db.get_memory_candidates(nick)
            candidate_facts = [c.fact for c in existing_candidates]
            snapshot_memory_ids = tuple(r.id for r in existing_rows)
            snapshot_candidate_ids = tuple(c.id for c in existing_candidates)

            def _extract_memories_bg() -> None:
                # Short-circuit at the top — extraction may have been
                # queued before die() but not yet started.
                if self._llm_executor.closing:
                    return
                try:
                    extraction = self.llm_service.extract_memories(
                        nick,
                        channel,
                        user_text,
                        assistant_response,
                        existing_facts,
                        candidate_facts,
                    )
                    if not extraction.add and not extraction.reinforce:
                        self._prune_stale_memory_candidates(nick)
                        return

                    # Race protection: abort if memories or candidates
                    # changed during the LLM call. The reinforce indices
                    # reference our candidate snapshot, so a stale list
                    # would mis-target rows.
                    current = self.db.get_memories(nick)
                    current_candidates = self.db.get_memory_candidates(nick)
                    if (
                        tuple(r.id for r in current) != snapshot_memory_ids
                        or tuple(c.id for c in current_candidates) != snapshot_candidate_ids
                    ):
                        log.info(
                            "Memory extraction for %s aborted: rows changed",
                            nick,
                        )
                        return

                    threshold = self.registryValue("memoryPromotionThreshold")
                    promoted: list[str] = []
                    current_count = len(current)

                    # Reinforce — bump existing candidates, promote any
                    # that cross the threshold.
                    for idx in extraction.reinforce:
                        if not 0 <= idx < len(existing_candidates):
                            continue
                        cand = existing_candidates[idx]
                        new_mentions = self.db.reinforce_memory_candidate(cand.id, nick)
                        if new_mentions == 0:
                            continue
                        if new_mentions >= threshold and current_count < max_memories:
                            self.db.save_memory(nick, cand.fact, cand.source_channel)
                            self.db.delete_memory_candidate(cand.id, nick)
                            promoted.append(cand.fact)
                            current_count += 1

                    # Add — new candidates start at mentions=1 (or promote
                    # immediately if threshold == 1).
                    for fact in extraction.add:
                        if threshold <= 1 and current_count < max_memories:
                            self.db.save_memory(nick, fact, channel)
                            promoted.append(fact)
                            current_count += 1
                        else:
                            self.db.add_memory_candidate(nick, fact, channel)

                    self._prune_stale_memory_candidates(nick)

                    if not promoted:
                        return

                    # Trigger cleanup if counter reaches interval — only
                    # promotions count toward the cleanup cadence, since
                    # they're what changes the durable memory set.
                    cleanup_interval = self.registryValue("memoryCleanupInterval")
                    if cleanup_interval > 0:
                        count = self.db.increment_memory_saves(nick)
                        if count >= cleanup_interval:
                            # Re-check closing before the cleanup pass:
                            # extraction itself can take seconds.
                            if self._llm_executor.closing:
                                return
                            self.db.reset_memory_saves(nick)
                            self._run_memory_cleanup(nick, channel)

                except Exception:
                    log.exception("Memory extraction failed for %s", nick)

            event_name = f"llm_memory_{uuid.uuid4().hex[:8]}"

            def _enqueue() -> None:
                if self._llm_executor.closing:
                    return
                self._llm_executor.submit(f"memory_extract:{nick}", _extract_memories_bg)

            schedule.addEvent(_enqueue, time.time() + 0.1, name=event_name)

        except Exception:
            log.exception("Memory extraction scheduling failed for %s", nick)

    def _prune_stale_memory_candidates(self, nick: str) -> None:
        """Drop candidates whose last_seen is older than the configured TTL.

        TTL of 0 disables decay (candidates linger until promoted or the
        admin clears them).
        """
        ttl_days = self.registryValue("memoryCandidateTTLDays")
        if ttl_days <= 0:
            return
        cutoff = time.time() - (ttl_days * 86400)
        try:
            self.db.prune_memory_candidates(nick, cutoff)
        except Exception:
            log.exception("Memory candidate pruning failed for %s", nick)

    def _run_memory_cleanup(self, nick: str, channel: str) -> ToolCallbackResult:
        """Run memory cleanup for a user. Returns a ToolCallbackResult."""
        from .assistant import ToolCallbackResult as _ToolCallbackResult

        snapshot = self.db.get_memories(nick)
        if len(snapshot) < 2:
            return _ToolCallbackResult(True, "Not enough memories to clean up.")

        before_count = len(snapshot)
        result = self.llm_service.cleanup_memories(nick, channel, snapshot)

        if result.error:
            log.warning("Memory cleanup failed for %s: %s", nick, result.error)
            short = result.error.split(":")[0] if ":" in result.error else result.error
            return _ToolCallbackResult(False, f"Cleanup failed ({short}). Try again later.")

        # Abort if memory rows changed during LLM call (race protection).
        # Compare row IDs — a delete+insert preserving count would otherwise
        # let cleanup mis-target indices.
        current = self.db.get_memories(nick)
        snapshot_ids = tuple(r.id for r in snapshot)
        current_ids = tuple(r.id for r in current)
        if current_ids != snapshot_ids:
            return _ToolCallbackResult(True, "Cleanup skipped — memories changed while processing.")

        # Apply drops
        dropped = 0
        for idx in result.drop:
            if 0 <= idx < len(snapshot):
                self.db.delete_memory(nick, snapshot[idx].id)
                dropped += 1

        # Apply merges: delete sources, insert merged fact
        merged = 0
        merged_sources = 0
        for entry in result.merge:
            sources = [snapshot[i] for i in entry.indices if 0 <= i < len(snapshot)]
            if not sources:
                continue
            oldest = min(sources, key=lambda s: s.created_at)
            for source in sources:
                self.db.delete_memory(nick, source.id)
            self.db.save_memory(nick, entry.text, oldest.source_channel)
            merged += 1
            merged_sources += len(sources)
        after_count = before_count - dropped - merged_sources + merged

        parts = [f"Before: {before_count}"]
        if dropped:
            parts.append(f"dropped: {dropped}")
        if merged:
            parts.append(f"merged: {merged_sources} → {merged}")
        parts.append(f"after: {after_count}")
        return _ToolCallbackResult(True, " | ".join(parts))

    def _usage_status(self, result: AssistantResult, *, delivered: bool) -> tuple[str, str]:
        """``(status, error_detail)`` for an assistant turn's usage row.

        ``timeout`` for a call that timed out (stashed for retry or not), then
        ``content_blocked`` / ``error`` from ``result.error``. A turn with no
        error that still delivered nothing is an ``error`` too: it was billed
        and the user got nothing for it.
        """
        detail = (result.error or "")[:200]
        # ``is True``, so a test's MagicMock result can't read as a timeout.
        if result.timed_out is True:
            return "timeout", detail
        if result.error is not None:
            if self._is_content_blocked_error(result.error):
                return "content_blocked", detail
            return "error", detail
        if not delivered:
            return "error", "not delivered"
        return "success", ""

    def _store_context_and_log_usage(
        self,
        caller: Caller,
        command: str,
        text: str,
        response: str,
        result: AssistantResult,
        irc: callbacks.Irc,
        *,
        delivered: bool = True,
    ) -> None:
        """Log API usage for a command, and store the turn if it was delivered.

        Shared between all commands (ask, code, draw).

        The usage row is written whether or not anything reached IRC: the
        provider billed the turn either way, and a turn that spent and then
        delivered nothing (an echo after the retry, an empty reply, a failed
        send) used to vanish from the ledger. History is written only for a
        delivered reply with no error.

        History and the usage row go under the caller's storage key and
        reply target; background memory extraction runs only for a
        personal key.

        Args:
            caller: Who asked, and where
            command: Command name ("ask" or "draw")
            text: Original user input
            response: Text to store in context (e.g. LLM response or
                ``"[Generated image: <url>]"``)
            result: Result with usage metadata
            irc: IRC connection instance
            delivered: Whether the reply reached IRC (the second value
                ``_dispatch_assistant_reply`` returns).
        """
        nick, channel = caller.identity.key, caller.reply_target
        stored = delivered and result.error is None
        if stored and self._get_context_enabled(channel):
            ctx_cfg = self._get_context_config(channel)
            self.context.add_message(nick, channel, Role.USER, text, config=ctx_cfg)
            self.context.add_message(nick, channel, Role.ASSISTANT, response, config=ctx_cfg)
            # The shared buffer is the channel's; a PM never writes to it
            # (see _gather_history).
            if caller.channel is not None:
                shared = caller.channel
                self.context.add_channel_message(
                    shared, caller.identity.raw_nick, Role.USER, text, config=ctx_cfg
                )
                self.context.add_channel_message(
                    shared, irc.nick, Role.ASSISTANT, response, config=ctx_cfg
                )

        status, error_detail = self._usage_status(result, delivered=delivered)
        self.db.log_usage(
            nick,
            channel,
            command,
            result.model,
            result.prompt_tokens,
            result.completion_tokens,
            result.cost,
            prompt=text,
            status=status,
            error_detail=error_detail,
        )

        # Schedule background memory extraction for eligible commands. Only for
        # a personal key: memories are read by it alone, so a nick-keyed fact
        # is a wasted LLM call (and could land on someone else's account key).
        personal_key = caller.identity.personal_key
        if command in _MEMORY_COMMANDS and stored and personal_key:
            self._schedule_memory_extraction(personal_key, channel, text, result.content)

    def ask(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
        text: str,
    ) -> None:
        """<question>

        Ask the AI a question. Supports conversation context (follow-up questions)
        and vision (include image URLs in your question).

        Examples:
          @ask What is the capital of France?
          @ask Describe this: https://example.com/image.jpg
          @ask And what about Germany?  (follow-up using context)
        """
        # Skip ZNC playback messages
        if self._is_old_message(msg):
            return

        pf = self._run_preflight(
            irc,
            msg,
            text,
            "ask",
            require_account=False,
        )
        if pf.blocked:
            return

        self._dispatch_with_verse_routing(irc, msg, text, pf, entry_route="ask")

    ask = wrap(ask, [("checkCapability", "llm.ask"), "text"])

    def rp(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
        text: str,
    ) -> None:
        """<text> | on | off

        Roleplay: speak or act AS your avatar in the channel's shared story —
        an in-character verse turn (persona, scene, narrative tools). A bare
        mention of canon (a character name, the channel's keyword) no longer
        enters this mode on its own; it only grounds a normal reply in canon.

        - ``@rp <text>`` — one in-character turn.
        - ``@rp on`` — sticky mode: your plain messages are treated as roleplay
          turns until ``@rp off`` (or after a spell of silence). Slip a single
          message OOC with a leading ``//``.
        - ``@rp off`` — leave sticky mode.

        Needs an avatar in the verse and the llm.verse capability; without an
        avatar it degrades to a normal, canon-grounded chat reply.

        Example:
          @rp Archie kicks the door open and bellows for the lads
        """
        if self._is_old_message(msg):
            return
        # Share @ask's rate-limit config (no separate rp* limit keys); the
        # entry_route tag still distinguishes rp in traces.
        pf = self._run_preflight(irc, msg, text, "ask", require_account=False)
        if pf.blocked:
            return
        toggle = text.strip().lower()
        if toggle in ("on", "off"):
            self._roleplay_sticky_set(pf, toggle == "on")
            if toggle == "on":
                irc.reply(
                    "Roleplay mode ON — just talk and I'll answer in character. "
                    "`@rp off` to stop; a leading `//` slips one message OOC.",
                    prefixNick=False,
                )
            else:
                irc.reply("Roleplay mode off — back to normal chat.", prefixNick=False)
            return
        self._dispatch_with_verse_routing(
            irc, msg, text, pf, entry_route="rp_command", force_roleplay=True
        )

    rp = wrap(rp, [("checkCapability", "llm.verse"), "text"])

    def _dispatch_with_verse_routing(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        text: str,
        preflight: PreflightResult,
        *,
        entry_route: str,
        force_roleplay: bool = False,
    ) -> None:
        """Look up a VerseRoute for the (channel, nick, account, text) and
        dispatch to ``_ask_impl`` with the verse overrides applied when one
        matches; otherwise fall through to the chat path. Used from every
        call site that turns user-addressed text into an assistant request
        (the explicit ``@ask`` command, ``invalidCommand`` for bare
        ``vibebot foo`` text, and ``_route_addressed_to_assistant`` for
        nick-comma-prefixed text). Without this on every entry point a
        verse-enabled channel still falls back to the chat profile for any
        message that isn't routed through ``@ask``, so verse_record never
        fires and the canon goes unrecorded.

        ``force_roleplay`` (the @rp command) enters verse roleplay mode
        explicitly; sticky @rp and the ambient-prose promotion below set it for
        a turn too. Otherwise a canon mention only injects facts into the chat
        turn (``verse_context`` below) — it no longer arms roleplay."""
        # Fire the typing indicator the moment we know preflight passed —
        # before any DB work (verse route lookup, history fetch, memory
        # fetch, executor permit acquisition). Otherwise the user waits
        # several seconds before "is composing" appears.
        caller = preflight.caller
        channel = caller.reply_target
        stop_typing = self.llm_service._begin_typing(irc, msg)
        try:
            # Slice 3: sticky @rp promotes ambient "just talk" messages to
            # roleplay turns while a session is live (explicit @ask/@rp untouched).
            if (
                not force_roleplay
                and entry_route in _AMBIENT_ENTRY_ROUTES
                and self._roleplay_sticky_active(preflight)
            ):
                force_roleplay = True
            # OOC opt-out (((like this)) or a leading //) means "skip the verse
            # engine for this message". _verse_route_for enforces that for
            # roleplay, but the two ambient OUTPUT paths below run before/around
            # it, so they have to honour the marker themselves — otherwise
            # `// illustrate the lads` still spends a whole storybook. Gated on
            # verseEnabled because these markers only carry OOC meaning on a
            # verse channel; elsewhere they are ordinary text.
            # Verse is per channel: a PM has no verse store to read or grow.
            verse_enabled = not caller.is_private and self.registryValue("verseEnabled", channel)
            ooc = bool(verse_enabled and is_ooc(text))
            # Prose-first ambient story, the DEFAULT for a canon mention: any
            # message that references canon becomes a multi-paragraph INLINE
            # tale in the avatar's voice (the verse completion) — no image,
            # questions included. One-shot: this promotes only THIS turn; it
            # does not arm sticky @rp. Only an explicit draw/illustrate request
            # keeps its own path (image / storybook), both canon-grounded.
            if (
                not force_roleplay
                and not ooc
                and entry_route in _AMBIENT_ENTRY_ROUTES
                and self._ambient_inline_story(preflight, text)
            ):
                force_roleplay = True
            route = self._verse_route_for(
                channel,
                caller.identity.raw_nick,
                caller.identity.account,
                text,
                msg.prefix,
                force_roleplay=force_roleplay,
            )
            if route is None:
                # Strip the OOC marker so the chat model receives a clean prompt
                # instead of the literal ((...)) / //.
                if ooc:
                    text = strip_ooc(text)
                # Explicit "illustrate"/comic mention → grounded illustrated
                # STORYBOOK (an image page). Only for "just talk" turns (not
                # @ask/@rp) that ask to be illustrated, reference canon, are not
                # OOC-opted-out, and pass the @story spend gates. Every other
                # canon mention became an inline prose tale above
                # (force_roleplay); a draw request returns None here and falls
                # through to chat (drawn), still verse-grounded.
                if verse_enabled and not ooc and entry_route in _AMBIENT_ENTRY_ROUTES:
                    brief = self._ambient_storybook_brief(msg, preflight, text)
                    if brief is not None:
                        # A brief means an authenticated account (its spend
                        # gate), so the personal key is set.
                        personal_key = caller.identity.personal_key
                        persona = (
                            (self.db.get_avatar_persona(personal_key) or "") if personal_key else ""
                        )
                        self._submit_storybook_job(
                            caller=caller,
                            persona=persona,
                            brief=brief,
                            max_images=int(
                                self.registryValue("verseStorybookMaxImages", channel) or 3
                            ),
                        )
                        return
                # Verse-enabled channel + non-opted-in speaker: advertise the
                # verse tool *schemas* anyway so the channel's tool surface is
                # byte-identical across all speakers. Invocations land on
                # denial handlers (see ``_ask_impl``). Without this branch the
                # tools list flips between roughly 15 and 23 entries depending
                # on opt-in state, which fragments xAI's automatic prompt
                # cache per-user instead of per-channel.
                if verse_enabled:
                    max_actors = self.registryValue("verseAutoEntityMaxNamesPerCall", channel)
                    verse_specs = make_verse_tool_specs(
                        max_actors=max_actors,
                        storybook=bool(self.registryValue("verseStorybookEnabled", channel)),
                    )
                    verse_context = self._verse_context_for(preflight, text, for_chat=True)
                    # A real-world question with NO canon reference (verse_context
                    # is None) is a genuine ask, not a story cue — answer it
                    # straight instead of letting the "tell the tallest tales"
                    # overlay fictionalise it (e.g. "who got rid of school milk"
                    # → Margaret Thatcher, not invented Stinky Lads lore). A
                    # question that DOES pull canon keeps the in-world voice.
                    overlay_override = (
                        _FACTUAL_CHAT_OVERLAY
                        if verse_context is None and self._looks_like_question(text)
                        else None
                    )
                    self._ask_impl(
                        irc,
                        msg,
                        text,
                        preflight,
                        entry_route=entry_route,
                        extra_tools_override=verse_specs,
                        verse_context=verse_context,
                        verse_record_handler=self._verse_chat_record_handler(preflight),
                        overlay_override=overlay_override,
                    )
                    return
                self._ask_impl(irc, msg, text, preflight, entry_route=entry_route)
                return
            # Per-channel verse model override. Empty string falls back to
            # assistantModel inside ``_ask_impl``. Useful when the channel's
            # assistantModel is a reasoning model that hard-caps verse output
            # at ~120 visible tokens regardless of prompt — point ``verseModel``
            # at a non-reasoning model (e.g. gemini-flash-latest) for richer
            # long-form scenes without affecting chat-mode behavior.
            verse_model = self.registryValue("verseModel", channel) or None
            if verse_model is None and channel not in self._verse_model_warned:
                self._verse_model_warned.add(channel)
                self.log.warning(
                    "verse turn on channel=%s has empty verseModel; falling back to "
                    "assistantModel — set a non-reasoning verseModel or verse prose may be "
                    "cratered by a reasoning model",
                    channel,
                )
            self._ask_impl(
                irc,
                msg,
                text,
                preflight,
                entry_route=entry_route,
                system_prompt_override=route.system_prompt,
                extra_tools_override=route.tools,
                profile_override=PROFILE_VERSE,
                verse_route=route,
                model_override=verse_model,
            )
        finally:
            stop_typing()

    def _ask_impl(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        text: str,
        pf: PreflightResult,
        *,
        entry_route: str = "ask",
        system_prompt_override: str | None = None,
        extra_tools_override: list[dict] | None = None,
        profile_override: str | None = None,
        verse_route: VerseRoute | None = None,
        model_override: str | None = None,
        verse_context: str | None = None,
        verse_record_handler: Callable[[dict], _VerseToolResult] | None = None,
        overlay_override: str | None = None,
    ) -> None:
        """Core ask logic, separated so invalidCommand can reuse without double-preflight.

        Optional keyword-only overrides for verse routing (C7c/C7d):
        - ``system_prompt_override``: when set, replaces the normal
          ``assistantSystemPrompt`` personality overlay entirely.
        - ``extra_tools_override``: tool specs appended to the profile tool list.
        - ``profile_override``: when set, overrides the profile selection
          (e.g. PROFILE_VERSE to bypass the token cap).
        - ``verse_route``: when set, verse tool handlers are built and merged
          into extra_handlers so the assistant loop can dispatch them (C7d).
        - ``verse_context``: facts-only canon block appended to the chat overlay
          (retrieval side of the canon layer). ADDITIVE — unlike
          ``system_prompt_override`` it keeps the personality overlay and layers
          reference lore after it, so a normal answer is grounded in canon
          without assuming the avatar persona.
        - ``overlay_override``: REPLACES the channel ``assistantSystemPrompt``
          personality overlay (e.g. swap a "tell the tallest tales" overlay for
          ``_FACTUAL_CHAT_OVERLAY`` so a real-world question gets a straight
          answer, not invented lore). Distinct from ``system_prompt_override``,
          which APPENDS a verse scene to the overlay; here we substitute it.
        """
        caller = pf.caller
        nick, channel = caller.identity.key, caller.reply_target
        effective_profile = profile_override or PROFILE_CHAT
        request_context = self._build_request_context(
            irc,
            msg,
            pf,
            entry_route=entry_route,
            profile=effective_profile,
        )

        with self._trace_request("ask", nick, channel):
            # Detect images for vision
            images = self.llm_service.detect_images(text)

            history, channel_history = self._gather_history(caller)

            # Personal data: read only through the personal key.
            personal_key = caller.identity.personal_key
            memories = self._get_user_memories(personal_key) if personal_key else []
            user_instruction = self.db.get_instruction(personal_key) if personal_key else None

            # Personality overlay assembly.
            #
            # Default (chat): channel ``assistantSystemPrompt``, optionally
            # prefixed with the user's persistent @instruct. The structural
            # framework (plain-text rules, tool-behavior rules) is layered in
            # by ``assistant_completion``.
            #
            # Verse: the override is just avatar identity + scene + recent
            # canon. Empirically dropping the channel's ``assistantSystemPrompt``
            # ("tell the tallest tales", "go mental") here cratered output
            # length on #afternet — grok happily produced 600+ tokens in chat
            # under the same model with that overlay, but only ~150 tokens in
            # verse without it. The energy/length pump comes from the
            # personality overlay, not the framework. So we PREPEND the
            # channel overlay (and @instruct) to the verse scene context;
            # the framework's verse-mode rules still apply on top.
            # The user's @instruct rides as user-role data (passed as
            # user_instruction below), NOT prepended to the system overlay —
            # a user request must not pose as system/developer authority. The
            # channel overlay (ask_prompt) and verse override are unchanged, so
            # the verse energy/length pump is preserved.
            ask_prompt = self.registryValue(PROFILES[effective_profile].overlay_setting, channel)
            # A real-world question with no canon reference: substitute the
            # channel personality overlay ("tell the tallest tales, go mental")
            # with a factual one so the answer is true, not invented lore. Never
            # combined with a verse scene (system_prompt_override), which is a
            # roleplay turn — this is the plain chat path only.
            if overlay_override is not None:
                ask_prompt = overlay_override
            if system_prompt_override is not None:
                parts: list[str] = []
                if ask_prompt:
                    parts.append(ask_prompt)
                parts.append(system_prompt_override)
                effective_prompt = "\n\n".join(parts)
            else:
                effective_prompt = ask_prompt
            # Retrieval side of the canon layer: append facts-only lore AFTER the
            # personality overlay (never replacing it), so a canon mention grounds
            # a normal chat answer without the roleplay persona. Off on roleplay
            # turns (system_prompt_override already carries the full scene).
            if verse_context and system_prompt_override is None:
                effective_prompt = "\n\n".join(p for p in [effective_prompt, verse_context] if p)

            with self._allow_concurrent(), self._llm_executor.permit():
                request_text = text
                if images:
                    # Clean prompt by removing image URLs
                    for img in images:
                        request_text = request_text.replace(img, "").strip()

                bridge_trace: list = []
                bridge_schemas, bridge_handlers = self._build_bridge_tool(
                    irc, msg, channel, trace=bridge_trace
                )
                bridge_debug = bool(
                    bridge_schemas and self.registryValue("bridgeDebugInChannel", channel)
                )
                # irc_lookup rides with the bridge extras: it needs the live
                # ``irc`` the same way run_limnoria_command does, and like
                # them it is advertised to every speaker so the channel's
                # cacheable prompt prefix stays byte-stable.
                if self.registryValue("ircLookupEnabled", channel):
                    lookup_schemas, lookup_handlers = self._build_irc_lookup_tool(
                        irc, caller.channel or ""
                    )
                    bridge_schemas = [*(bridge_schemas or []), *lookup_schemas]
                    bridge_handlers = {**(bridge_handlers or {}), **lookup_handlers}
                if self.registryValue("reactEnabled", channel):
                    react_schemas, react_handlers = self._build_react_tool(irc, msg, caller)
                    bridge_schemas = [*(bridge_schemas or []), *react_schemas]
                    bridge_handlers = {**(bridge_handlers or {}), **react_handlers}
                if self.registryValue("redactEnabled", channel):
                    redact_schemas, redact_handlers = self._build_redact_tool(irc, msg, caller)
                    bridge_schemas = [*(bridge_schemas or []), *redact_schemas]
                    bridge_handlers = {**(bridge_handlers or {}), **redact_handlers}
                if self.registryValue("notifyOnlineEnabled", channel):
                    notify_schemas, notify_handlers = self._build_notify_tool(irc, msg, caller)
                    bridge_schemas = [*(bridge_schemas or []), *notify_schemas]
                    bridge_handlers = {**(bridge_handlers or {}), **notify_handlers}
                if self.registryValue("memeEnabled", channel):
                    meme_schemas, meme_handlers = self._build_meme_tool(msg)
                    bridge_schemas = [*(bridge_schemas or []), *meme_schemas]
                    bridge_handlers = {**(bridge_handlers or {}), **meme_handlers}
                if self.registryValue("auroraEnabled", channel):
                    aurora_schemas, aurora_handlers = self._build_aurora_tool()
                    bridge_schemas = [*(bridge_schemas or []), *aurora_schemas]
                    bridge_handlers = {**(bridge_handlers or {}), **aurora_handlers}
                # Combine bridge tools with any verse tools from the route.
                bridge_list = list(bridge_schemas) if bridge_schemas else []
                verse_list = list(extra_tools_override) if extra_tools_override else []
                extra_tools = (bridge_list + verse_list) or None

                # C7d: merge verse handlers into extra_handlers so the
                # assistant_completion loop can dispatch verse tool calls
                # in-flight rather than passing them to the generic executor.
                #
                # When verse tool *schemas* are advertised on a verse-enabled
                # channel but the caller hasn't opted in (verse_route is None
                # while extra_tools_override carries verse specs), wire denial
                # handlers so the model gets a clean rejection. Advertising
                # the same tool surface to every speaker keeps the channel's
                # cacheable prefix byte-stable across opted-in/non-opted-in
                # users — the cohort split that was costing prompt-cache hits.
                if verse_route is not None:
                    verse_handlers = self._build_verse_handlers_for_route(channel, verse_route)
                    combined_handlers: dict | None = {
                        **(bridge_handlers or {}),
                        **verse_handlers,
                    }
                    # verse_edit (gated): overlay a live handler bound to the
                    # TRIGGERING user's prefix so the llm.verse.edit capability
                    # check reflects who drove this turn, not the channel's
                    # active avatar. The tool spec is always advertised on a
                    # verse route (make_verse_tool_specs), so the handler must
                    # always be present here — an unauthorized caller is refused
                    # inside dispatch_verse_edit, not by withholding the tool.
                    combined_handlers["verse_edit"] = self._verse_edit_handler(
                        msg=msg,
                        store=verse_route.store,
                        account=caller.identity.account,
                    )
                    # Storybook tool (gated): overlay a live handler that can
                    # capture irc/msg/channel/account/persona. Only wired
                    # when the per-channel flag is on; otherwise the spec isn't
                    # advertised and this key is never reached. The persona is
                    # the clean character voice (NOT the system_prompt overlay).
                    if self.registryValue("verseStorybookEnabled", channel):
                        persona = (
                            (self.db.get_avatar_persona(personal_key) or "") if personal_key else ""
                        )
                        combined_handlers["verse_storybook"] = self._storybook_handler(
                            irc=irc,
                            msg=msg,
                            caller=caller,
                            persona=persona,
                        )
                elif verse_list:
                    denial_handlers = make_verse_denial_handlers(verse_list)
                    combined_handlers = {
                        **(bridge_handlers or {}),
                        **denial_handlers,
                    }
                    # Slice 2: overlay a LIVE verse_record over its denial handler
                    # so an opted-in avatar's ordinary chat turn can save canon
                    # (gated by verseChatRecordEnabled; None otherwise). All other
                    # verse tools stay denied on the chat path — only recording is
                    # freed, not the roleplay actions.
                    if verse_record_handler is not None:
                        combined_handlers["verse_record"] = verse_record_handler
                else:
                    combined_handlers = bridge_handlers

                # Typing keepalive is started in
                # ``_dispatch_with_verse_routing`` so it covers the verse
                # route lookup and history/memory fetches that run before
                # this point. ``manage_typing=False`` keeps the
                # service-level layer from clobbering it.
                result = self.llm_service.assistant_completion(
                    request_text,
                    **request_context.completion_kwargs(),
                    db=self.db,
                    context=self.context,
                    bot_nick=irc.nick,
                    images=images,
                    history=history,
                    channel_history=channel_history,
                    irc=irc,
                    msg=msg,
                    memories=memories,
                    user_instruction=user_instruction,
                    system_prompt=effective_prompt,
                    model_override=model_override,
                    search_fn=lambda q: self.llm_service.search_completion(q, channel=channel),
                    fetch_fn=lambda u: self.llm_service.url_completion(u, channel=channel),
                    draw_fn=lambda p, **kw: self._draw_for_assistant(irc, msg, p, **kw),
                    cleanup_fn=lambda n: self._run_memory_cleanup(n, channel),
                    extra_tools=extra_tools,
                    extra_handlers=combined_handlers,
                    manage_typing=False,
                    exclude_tools=self._pending_task_excludes(channel),
                    **self._pending_task_fns(caller=caller, irc=irc, msg=msg),
                )

                response = result.content

                # Optional in-channel debug footer listing bridge tool calls.
                if bridge_debug and bridge_trace:
                    footer = self._format_bridge_debug_footer(bridge_trace)
                    if footer:
                        response = f"{response}\n{footer}" if response else footer

                self._reply_and_book(
                    irc,
                    msg,
                    result,
                    caller=caller,
                    command="ask",
                    text=text,
                    response=response,
                    suppress_reminder_mutations=True,
                )

    def draw(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
        text: str,
    ) -> None:
        """<prompt>

        Generate an image from a text description.

        Examples:
          @draw A sunset over mountains in watercolor style
          @draw A cyberpunk cityscape at night
        """
        # Skip ZNC playback messages
        if self._is_old_message(msg):
            return
        off = self.llm_service.draw_unavailable_reason(msg.channel)
        if off:
            irc.reply(off)
            return

        pf = self._run_preflight(
            irc,
            msg,
            text,
            "draw",
            require_account=True,
        )
        if pf.blocked:
            return
        caller = pf.caller
        nick, channel = caller.identity.key, caller.reply_target

        # Typing fires immediately after preflight so users see "is composing"
        # before the subject-research call / history fetch / executor permit /
        # image generation latency stack up. The overlay assembly sits inside
        # the try for that reason: @draw has nothing to hide a research round
        # trip behind, so the indicator goes up first.
        stop_typing = self.llm_service._begin_typing(irc, msg)
        try:
            # Verse grounding: if the prompt references canon, layer the
            # facts-only lore block onto the draw overlay (personality-overlay
            # slot; the draw framework/tools are still added downstream by
            # assistant_completion) so "@draw the stinky lads" depicts the
            # canon cast, not literal words.
            overlay_parts: list[str] = []
            verse_ctx = self._verse_context_for(pf, text)
            if verse_ctx:
                overlay_parts.append(self._verse_grounded_overlay(PROFILE_DRAW, channel, verse_ctx))
            # Subject research: canon covers what the channel invented, this
            # covers what the world already contains. Second, so canon is
            # stated first — the block itself says canon wins on overlap.
            with self._allow_concurrent(), self._llm_executor.permit():
                dossier_block = self._subject_dossier_for(nick, channel, text)
            if dossier_block:
                overlay_parts.append(dossier_block)
            # None when nothing is referenced and nobody real is named → the
            # default draw prompt, unchanged.
            draw_system_prompt = "\n\n".join(overlay_parts) or None

            request_context = self._build_request_context(
                irc,
                msg,
                pf,
                entry_route=PROFILE_DRAW,
                profile=PROFILE_DRAW,
            )

            with self._trace_request("draw", nick, channel):
                history, channel_history = self._gather_history(
                    caller,
                    max_age_seconds=self.registryValue("drawContextMaxAgeSeconds", channel),
                )

                with self._allow_concurrent(), self._llm_executor.permit():
                    result = self.llm_service.assistant_completion(
                        text,
                        **request_context.completion_kwargs(),
                        db=self.db,
                        context=self.context,
                        bot_nick=irc.nick,
                        history=history,
                        channel_history=channel_history,
                        irc=irc,
                        msg=msg,
                        memories=[],
                        system_prompt=draw_system_prompt,
                        draw_fn=lambda p, **kw: self._draw_for_assistant(irc, msg, p, **kw),
                        manage_typing=False,
                        **self._pending_task_fns(caller=caller, irc=irc, msg=msg),
                    )

                    self._reply_and_book(
                        irc,
                        msg,
                        result,
                        caller=caller,
                        command="draw",
                        text=text,
                        response=result.content,
                    )
        finally:
            stop_typing()

    draw = wrap(draw, [("checkCapability", "llm.draw"), "text"])

    def _meme_catalog(self) -> meme.MemeCatalog | None:
        """The bundled template catalog plus topic tags and aliases. None
        only when ``memes/templates.json`` is missing from the install."""
        catalog = meme.bundled_catalog()
        if catalog is None:
            self.log.warning("meme: templates unavailable: %s missing", meme.CATALOG_PATH)
            return None
        catalog = catalog.with_keywords(meme_topics.TOPICS)
        aliases = meme.parse_aliases(self.registryValue("memeAliases") or [])
        return catalog.with_aliases(aliases) if aliases else catalog

    def _meme_names(self, query: str, *, has_lines: bool) -> bool:
        """True when the user chose the template, so the picker stays out.

        An image URL or an exact id/name always counts. A fuzzy match
        ("willy wonka" → Condescending Wonka) counts only when captions came
        with it: "@meme y'all got any more of them flamethrowers" also
        fuzzy-matches yallgot, and it is a request, not a name.
        """
        if meme.is_http_url(query.strip()):
            return True
        catalog = self._meme_catalog()
        if catalog is None:
            return False
        if catalog.exact(query) is not None:
            return True
        return has_lines and catalog.resolve(query) is not None

    def _make_meme(
        self,
        msg: IrcMsg,
        template_query: str,
        lines: list[str],
        options: meme.MemeOptions | None = None,
    ) -> tuple[str | None, str]:
        """Resolve, render, host. ``(url, error)``; one is set.

        Shared by @meme and the make_meme tool so an unknown name or a wrong
        caption count reads the same in both places.

        Writes one $0 usage row per render, under model ``meme-render``:
        free, but media generation is attributable, and @usage should count a
        meme the way it counts a draw. A resolver miss never renders and gets
        no row, same as a draw prompt that fails validation.
        """
        options = options or meme.MemeOptions()
        catalog = self._meme_catalog()
        if catalog is None:
            return None, _("Meme templates are unavailable right now.")
        plan = meme.plan_meme(catalog, template_query, lines, options)
        if plan.error or plan.lines is None or plan.template is None:
            return None, plan.error or _("Could not build that meme.")
        self.log.info("meme: template=%s lines=%s", plan.template.id, len(lines))
        # A template with picture slots gets generated pictures masked into
        # its blanks before the captions go on; any other template keeps the
        # whole-image edit below.
        pictures: list[Image.Image | None] = []
        picture_note = ""
        if options.draw and plan.template.slots:
            pictures, picture_note = self._slot_pictures(msg, plan.template, options.draw)
        hosted, error = self._render_meme(plan.template, plan.lines, options, pictures)
        try:
            self.db.log_usage(
                ircutils.nickFromHostmask(msg.prefix),
                Caller.from_msg(msg).reply_target,
                "meme",
                "meme-render",
                0,
                0,
                0.0,
                prompt=" | ".join([plan.template.id, *lines])[:200],
                status="success" if hosted else "error",
                error_detail=error[:200],
            )
        except Exception:
            self.log.exception("meme usage logging failed")
        if not hosted:
            return None, error
        if picture_note:
            return f"{hosted} — {picture_note}", ""
        if options.draw and not plan.template.slots:
            return self._edit_meme(msg, hosted, options.draw), ""
        return hosted, ""

    # Parallel picture generations for one meme; the starter pack has four.
    _MEME_PICTURE_WORKERS = 4

    def _slot_pictures(
        self, msg: IrcMsg, template: meme.MemeTemplate, draw: str
    ) -> tuple[list[Image.Image | None], str]:
        """One generated picture per slot, and a note if any failed.

        Each slot's picture is generated at the aspect ratio nearest its
        box, in parallel, and booked as its own ``meme`` usage row under the
        image model. A failed slot stays blank and the note says why: the
        captions are the meme, the pictures are extras.
        """
        assert template.image is not None
        base = meme_render.load_image(template.image)
        drawings = meme.slot_drawings(draw, len(template.slots))
        # One generation per distinct description, reused for every slot
        # that asks for it: "same" must show the same picture twice.
        first_slot: dict[str, int] = {}
        for index, drawing in enumerate(drawings):
            first_slot.setdefault(drawing, index)

        def one(drawing: str) -> MemePicture:
            width, height = meme_render.slot_size(base, template.slots[first_slot[drawing]])
            ratio = meme.nearest_aspect_ratio(width, height)
            return self.llm_service.image_generate(drawing, aspect_ratio=ratio)

        unique = list(first_slot)
        workers = min(self._MEME_PICTURE_WORKERS, len(unique))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = dict(zip(unique, pool.map(one, unique), strict=True))
        made: dict[str, Image.Image | None] = {}
        errors: list[str] = []
        for drawing, result in results.items():
            try:
                self.db.log_usage(
                    ircutils.nickFromHostmask(msg.prefix),
                    Caller.from_msg(msg).reply_target,
                    "meme",
                    result.model,
                    0,
                    0,
                    result.cost,
                    prompt=drawing[:200],
                    status="error" if result.error else "success",
                    error_detail=(result.error or "")[:200],
                )
            except Exception:
                self.log.exception("meme usage logging failed")
            picture = None
            if result.data:
                try:
                    picture = meme_render.load_image(result.data)
                except Exception:
                    self.log.exception("meme: generated picture unreadable")
            if picture is None:
                errors.append(result.error or "no picture")
            made[drawing] = picture
        self.log.info(
            "meme_pictures: template=%s slots=%s generated=%s failed=%s",
            template.id,
            len(template.slots),
            len(made),
            len(errors),
        )
        pictures = [made[drawing] for drawing in drawings]
        if not errors:
            return pictures, ""
        if len(errors) == len(made):
            return pictures, f"picture not added: {errors[0]}"
        return pictures, f"{len(errors)} of {len(made)} pictures not added: {errors[0]}"

    def _render_meme(
        self,
        template: meme.MemeTemplate,
        lines: list[str],
        options: meme.MemeOptions,
        pictures: list[Image.Image | None] | None = None,
    ) -> tuple[str | None, str]:
        """Draw the captions and host the PNG. ``(url, error)``; one is set.

        ``pictures`` go into the template's slots first, so the captions sit
        on top. A custom template's image is fetched through the same
        guarded downloader as any untrusted URL.
        """
        source: Path | bytes | None = template.image
        if template.background:
            fetched = self.llm_service._download_image_bytes(template.background)
            if fetched is None:
                return None, _("Could not fetch that image.")
            source = fetched[0]
        if source is None:
            return None, _("That template has no image.")
        try:
            image = meme_render.load_image(source)
            if pictures and template.slots:
                image = meme_render.insert_pictures(image, template.slots, pictures)
            png = meme_render.render(
                image,
                template.boxes,
                lines,
                font=options.font,
                layout_top=options.layout_top,
            )
        except Exception:
            self.log.exception("meme: render failed for %s", template.id)
            return None, _("Could not draw that meme.")
        hosted = self.llm_service._save_image_bytes(png, "png")
        return (hosted, "") if hosted else (None, _("Could not save that meme."))

    def _edit_meme(self, msg: IrcMsg, hosted: str, draw: str) -> str:
        """Send the captioned meme through the edit model; the line to post.

        The captions are already pixels, so a refusal costs the picture and
        not the meme: the reply is the captioned URL with a note. The edit's
        own row goes into the usage table under the edit model, beside the
        $0 render row, at what xAI reported (or the list price).
        """
        channel = Caller.from_msg(msg).reply_target
        result = self.llm_service.image_edit(hosted, draw, channel=channel)
        try:
            self.db.log_usage(
                ircutils.nickFromHostmask(msg.prefix),
                channel,
                "meme",
                result.model,
                0,
                0,
                result.cost,
                prompt=draw[:200],
                status="error" if result.error else "success",
                error_detail=(result.error or "")[:200],
            )
        except Exception:
            self.log.exception("meme usage logging failed")
        self.log.info("meme_edit: draw=%r -> %s", draw[:120], result.error or result.url)
        if result.error or not result.url:
            return f"{hosted} — picture not added: {result.error or 'no image'}"
        return result.url

    def _meme_or_infer(
        self,
        msg: IrcMsg,
        template_query: str,
        lines: list[str],
        options: meme.MemeOptions | None = None,
        *,
        named: bool = True,
        context: str | None = None,
    ) -> tuple[str | None, str]:
        """Render a named template, or let the picker choose one.

        The picker runs only when ``template_query`` names nothing the
        catalog knows — '@meme waiting for CI', or a miss with captions
        ('@meme how did you get so strong | ...'). When the picker declines
        and the user had ``named`` a template, the resolver's did-you-mean
        list follows its reason; a brief gets the reason alone.
        """
        if self._meme_names(template_query, has_lines=any(x.strip() for x in lines)):
            return self._make_meme(msg, template_query, lines, options)
        request = " | ".join(x for x in [template_query, *lines] if x)
        if options and options.draw:
            # The chat model's picture instruction is the subject of the
            # meme; without it the picker captioned a costume "My Meme".
            request = f"{request}\nPicture: {options.draw}"
        hosted, error = self._infer_meme(msg, request, options, context=context)
        if hosted is None and named:
            _hosted, resolver_error = self._make_meme(msg, template_query, lines, options)
            error = f"{error} {resolver_error}"
        return hosted, error

    def _infer_meme(
        self,
        msg: IrcMsg,
        request: str,
        options: meme.MemeOptions | None = None,
        *,
        context: str | None = None,
    ) -> tuple[str | None, str]:
        """Let the meme model pick the template and captions, then render.

        Runs only when the user named no template the catalog knows. The
        pick is validated by :func:`meme.parse_pick` before anything renders;
        the picker's own completion is booked as a ``meme`` usage row under
        its model, beside the $0 row the render writes. Success is
        ``url — id | line | line`` so the channel sees what was chosen and
        can redo it by hand.
        """
        catalog = self._meme_catalog()
        if catalog is None:
            return None, _("Meme templates are unavailable right now.")
        channel = Caller.from_msg(msg).reply_target
        pick = self.llm_service.meme_pick(
            request,
            catalog_brief=meme.catalog_brief(
                catalog,
                request,
                popular=meme.SEARCH_BRIEF_POPULAR,
                matches=meme.SEARCH_BRIEF_MATCHES,
            ),
            channel=channel,
            context=context,
            search=lambda query: meme.search_brief(catalog, query),
        )
        try:
            self.db.log_usage(
                ircutils.nickFromHostmask(msg.prefix),
                channel,
                "meme",
                pick.model,
                pick.prompt_tokens,
                pick.completion_tokens,
                pick.cost,
                prompt=request[:200],
                status="error" if pick.error else "success",
                error_detail=(pick.error or "")[:200],
            )
        except Exception:
            self.log.exception("meme usage logging failed")
        if pick.error:
            return None, pick.error
        choice = meme.parse_pick(pick.content, catalog)
        self.log.info("meme_pick: request=%r -> %s", request[:120], choice)
        if isinstance(choice, str):
            return None, choice
        options = options or meme.MemeOptions()
        if choice.draw and not options.draw:
            options = replace(options, draw=choice.draw)
        hosted, error = self._make_meme(msg, choice.template.id, choice.lines, options)
        if hosted is None:
            return None, error
        return f"{hosted} — {' | '.join([choice.template.id, *choice.lines])}", ""

    # --- aurora tool ---------------------------------------------------------

    # Feed cache lifetimes, shared by the tool and the announcer.
    _SW_RTSW_TTL = timedelta(minutes=2)
    _SW_OVATION_TTL = timedelta(minutes=10)
    _SW_KP_TTL = timedelta(hours=3)
    _SW_ALERTS_TTL = timedelta(minutes=5)
    _SW_CLOUDS_TTL = timedelta(hours=1)
    _SW_OVATION_STALE = timedelta(minutes=60)
    _SW_PLACE_HIT_TTL = 24 * 3600.0
    _SW_PLACE_MISS_TTL = 3600.0
    # Storm-conditions rule and the lowest watch level announced. Fixed by the
    # developer, not the operator.
    _SW_BZ_THRESHOLD_NT = -10.0
    _SW_BZ_HOURS = 3.0
    _SW_MIN_WATCH_LEVEL = 3
    _SW_OVATION_NOTE = (
        "poleward_max is the strongest aurora within 10° toward the pole; "
        "a stand-in for what is visible low on the horizon"
    )

    def _sw_place(self, query: str) -> spaceweather.Place | None:
        """Geocode with a cache: hits 24 h, misses 1 h, FetchError never."""
        key = query.strip().casefold()
        now = time.monotonic()
        with self._sw_lock:
            cached = self._sw_places.get(key)
            if cached is not None:
                at, place = cached
                ttl = self._SW_PLACE_HIT_TTL if place is not None else self._SW_PLACE_MISS_TTL
                if now - at < ttl:
                    return place
        place = spaceweather.geocode(query.strip())
        with self._sw_lock:
            self._sw_places[key] = (now, place)
        return place

    def _sw_cached[T](self, key: str, ttl: timedelta, now: datetime, loader: Callable[[], T]) -> T:
        """``loader()``'s value, reused while younger than ``ttl``.

        The lock is never held during ``loader()``, so two callers may both
        fetch on a cold key; the later store wins. Exceptions are not cached.
        """
        with self._sw_lock:
            cached = self._sw_cache.get(key)
        if cached is not None and now - cached[0] < ttl:
            return cached[1]
        value = loader()
        prefix = key.partition(":")[0] + ":" if ":" in key else None
        with self._sw_lock:
            if prefix is not None:
                # Per-place keys: drop this family's expired entries so the
                # dict stays bounded by the places asked about within ``ttl``.
                for k in [
                    k
                    for k, (at, _v) in self._sw_cache.items()
                    if k.startswith(prefix) and now - at >= ttl
                ]:
                    del self._sw_cache[k]
            self._sw_cache[key] = (now, value)
        return value

    def _sw_swpc(self, path: str, max_bytes: int = spaceweather.MAX_BYTES) -> Any:
        return spaceweather.fetch_json(spaceweather.SWPC + path, max_bytes=max_bytes)

    def _sw_ovation_grid(self, now: datetime) -> dict:
        return self._sw_cached(
            "ovation",
            self._SW_OVATION_TTL,
            now,
            lambda: self._sw_swpc(spaceweather.OVATION_PATH, spaceweather.OVATION_MAX_BYTES),
        )

    def _sw_samples(self, now: datetime) -> list[spaceweather.Sample]:
        def load() -> list[spaceweather.Sample]:
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="aurora-rtsw") as pool:
                mag = pool.submit(self._sw_swpc, spaceweather.MAG_PATH)
                wind = pool.submit(self._sw_swpc, spaceweather.WIND_PATH)
                return spaceweather.parse_rtsw(mag.result(), wind.result())

        return self._sw_cached("rtsw", self._SW_RTSW_TTL, now, load)

    def _sw_kp_rows(self, now: datetime) -> Any:
        return self._sw_cached(
            "kp", self._SW_KP_TTL, now, lambda: self._sw_swpc(spaceweather.KP_PATH)
        )

    def _sw_alerts(self, now: datetime) -> Any:
        return self._sw_cached(
            "alerts", self._SW_ALERTS_TTL, now, lambda: self._sw_swpc(spaceweather.ALERTS_PATH)
        )

    def _sw_cloud_series(
        self, place: spaceweather.Place, *, now: datetime
    ) -> list[tuple[datetime, int]]:
        """The place's cached 5-day hourly cloud series, keyed by lat/lon to 2 dp."""
        lat, lon = round(place.lat, 2), round(place.lon, 2)
        return self._sw_cached(
            f"clouds:{lat},{lon}",
            self._SW_CLOUDS_TTL,
            now,
            lambda: spaceweather.cloud_series(lat, lon),
        )

    def _sw_cloud_rows(
        self, place: spaceweather.Place, start: datetime, end: datetime, *, now: datetime
    ) -> list[tuple[datetime, int]]:
        """Hourly cloud cover within ``[start, end]``, from the place's cached series."""
        return spaceweather.hours_within(
            self._sw_cloud_series(place, now=now), start=start, end=end
        )

    def _aurora_solar_wind(self, now: datetime) -> dict:
        samples = self._sw_samples(now)
        newest_bz = max((s.ts for s in samples if s.bz is not None), default=None)
        if newest_bz is None or now - newest_bz > spaceweather.STALE_AFTER:
            raise spaceweather.FetchError("no Bz in the last 15 minutes")
        episode = spaceweather.advance_episode(
            None,
            samples,
            threshold_nt=self._SW_BZ_THRESHOLD_NT,
            hours=self._SW_BZ_HOURS,
            now=now,
        )
        bz = next((s.bz for s in reversed(samples) if s.bz is not None), None)
        speed = next((s.speed for s in reversed(samples) if s.speed is not None), None)
        return {
            "now": {
                "bz_nt": bz,
                "speed_kms": speed,
                "storm_minutes": (
                    int((now - episode.start).total_seconds() // 60) if episode else 0
                ),
                "lead_minutes": episode.lead_minutes if episode else None,
            }
        }

    def _aurora_ovation(self, found: spaceweather.Place, now: datetime) -> dict:
        hit = spaceweather.ovation_at(self._sw_ovation_grid(now), found.lat, found.lon)
        observed = _parse_iso_utc(hit["observed"])
        ovation: dict[str, Any] = {
            "overhead_pct": hit["overhead_pct"],
            "poleward_max_pct": hit["poleward_max_pct"],
            "poleward_max_lat": hit["poleward_max_lat"],
            "valid_until": hit["forecast_time"],
            "note": self._SW_OVATION_NOTE,
        }
        if observed is None or now - observed > self._SW_OVATION_STALE:
            ovation["stale"] = True
        return {"ovation": ovation}

    def _aurora_kp(self, now: datetime) -> dict:
        return {
            "kp_next_24h": [
                {"start": b["start"].isoformat(), "kp": b["kp"], "scale": b["scale"]}
                for b in spaceweather.parse_kp(self._sw_kp_rows(now), now=now)
            ]
        }

    def _aurora_alerts(self, now: datetime) -> dict:
        levels = spaceweather.effective_levels(
            spaceweather.parse_watches(self._sw_alerts(now)), today=now.date()
        )
        top = max(levels.values(), default=0)
        if top < 1:
            return {}
        return {
            "active_watch": {
                "level": f"G{top}",
                "dates": sorted(d.isoformat() for d, lvl in levels.items() if lvl == top),
            }
        }

    @staticmethod
    def _aurora_dark(
        found: spaceweather.Place, now: datetime
    ) -> tuple[dict, ZoneInfo, tuple[datetime, datetime] | None]:
        """(``dark`` payload, place zone, UTC window to read clouds for or None)."""
        try:
            zone = ZoneInfo(found.tz)
            tz_name = found.tz
        except (ZoneInfoNotFoundError, ValueError):
            zone, tz_name = ZoneInfo("UTC"), "UTC"
        local = now.astimezone(zone)
        # After dawn "tonight" is the coming night; before dawn, the night
        # still in progress.
        night = local.date()
        start, end = spaceweather.dark_window(found.lat, found.lon, night, tz_name)
        if local.hour < 12:
            prev = night - timedelta(days=1)
            p_start, p_end = spaceweather.dark_window(found.lat, found.lon, prev, tz_name)
            if p_start is not None and p_start <= now and (p_end is None or now < p_end):
                start, end = p_start, p_end
        if start is None:
            return {"never_dark": True, "tz": "local"}, zone, None
        dark: dict[str, Any] = {"from": start.astimezone(zone).strftime("%H:%M"), "tz": "local"}
        if end is None:
            dark["dark_until_dawn_not_found"] = True
        else:
            dark["until"] = end.astimezone(zone).strftime("%H:%M")
        return dark, zone, (start, end if end is not None else start + timedelta(hours=12))

    def _aurora_payload(self, place: str, *, now: datetime) -> dict:
        """Facts for the aurora tool. Each feed fails alone into ``unavailable``.

        The feeds are fetched concurrently on a per-call pool (not
        ``_llm_executor``: this already runs on one of its workers).
        """
        try:
            found = self._sw_place(place)
        except spaceweather.FetchError as exc:
            self.log.warning("aurora: geocode failed for %r: %s", place, exc)
            return {"error": "place lookup is unavailable right now"}
        if found is None:
            return {"error": "couldn't find that place"}

        dark, zone, window = self._aurora_dark(found, now)
        errors = (spaceweather.FetchError, spaceweather.InvalidPayload)
        feeds: dict[str, tuple[Callable[[], dict], tuple[type[Exception], ...]]] = {
            "solar_wind": (lambda: self._aurora_solar_wind(now), errors),
            "ovation": (
                lambda: self._aurora_ovation(found, now),
                (*errors, KeyError, TypeError, ValueError),
            ),
            "kp": (lambda: self._aurora_kp(now), errors),
            "alerts": (lambda: self._aurora_alerts(now), errors),
        }
        if window is not None:
            start, end = window

            def clouds() -> dict:
                cover = self._sw_cloud_rows(found, start, end, now=now)
                return {
                    "cloud_pct_while_dark": [
                        {"hour": ts.astimezone(zone).strftime("%H:00"), "pct": pct}
                        for ts, pct in cover
                    ]
                }

            feeds["clouds"] = (clouds, errors)

        out: dict[str, Any] = {"place": found.name, "lat": found.lat, "lon": found.lon}
        unavailable: list[str] = []
        with ThreadPoolExecutor(max_workers=5, thread_name_prefix="aurora") as pool:
            futures = {name: pool.submit(fn) for name, (fn, _caught) in feeds.items()}
            for name, fut in futures.items():
                if name == "clouds":
                    out["dark"] = dark
                try:
                    out.update(fut.result())
                except feeds[name][1] as exc:
                    self.log.warning("aurora: %s unavailable: %s", name, exc)
                    unavailable.append(name)
        if "dark" not in out:
            out["dark"] = dark

        if unavailable:
            out["unavailable"] = unavailable
        return out

    def _build_aurora_tool(self):
        """Build the ``aurora`` tool schema + handler."""
        from .assistant import ToolResult

        schema = {
            "type": "function",
            "function": {
                "name": "aurora",
                "description": (
                    "Aurora / northern-lights odds for a place: NOAA aurora probability, "
                    "current solar wind, Kp for the next 24 h, when it gets dark, and "
                    "cloud cover while dark. If the user names no place, use one from "
                    "what you know about them; if you know none, ask."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "place": {
                            "type": "string",
                            "description": "City or place name, e.g. 'Oshawa' or 'Tromsø'.",
                        }
                    },
                    "required": ["place"],
                },
            },
        }

        def handler(arguments: dict[str, Any]) -> ToolResult:
            place = arguments.get("place")
            if not isinstance(place, str) or not place.strip():
                return ToolResult(content=json.dumps({"error": "Give a place name."}))
            payload = self._aurora_payload(place, now=datetime.now(UTC))
            return ToolResult(content=json.dumps(payload))

        return [schema], {"aurora": handler}

    def _build_meme_tool(self, msg: IrcMsg):
        """Build the ``make_meme`` tool schema + handler.

        Same shape as :meth:`_build_irc_lookup_tool`. The model transcribes
        the template name the user said; resolution is :mod:`meme`'s job, and
        a miss comes back as an error with suggestions for the model to relay.
        """
        from .assistant import ToolResult

        schema = {
            "type": "function",
            "function": {
                "name": "make_meme",
                "description": (
                    "Make a meme and return the image URL. Two "
                    "modes: the user names a template and gives captions "
                    "(template + lines), or the user asks for a meme about "
                    "something (brief) and the tool picks the template and "
                    "writes the captions. Returns an error listing similar "
                    "template names when a named template is unknown."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "template": {
                            "type": "string",
                            "description": (
                                "The meme's name exactly as the user wrote it, e.g. "
                                "'drake', 'distracted boyfriend', 'this is fine' — or "
                                "an image URL the user pasted, to caption that image. "
                                "Do not choose one yourself."
                            ),
                        },
                        "lines": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "The captions in order, top to bottom / left to "
                                "right, verbatim from the user. Empty when the user "
                                "gave none."
                            ),
                        },
                        "brief": {
                            "type": "string",
                            "description": (
                                "What the meme should be about, in the user's words, "
                                "when they did not name a template. Leave template "
                                "empty in that case."
                            ),
                        },
                        "draw": {
                            "type": "string",
                            "description": (
                                "Only if the user asked to change the picture itself: "
                                "what to put in a blank photo area, or who/what to "
                                "swap in ('make the boyfriend a sysadmin'). For a "
                                "template with several picture areas, one description "
                                "per area separated by ' | '. Costs a draw per "
                                "picture; omit otherwise."
                            ),
                        },
                    },
                    "required": [],
                },
            },
        }

        def handler(arguments: dict[str, Any]) -> ToolResult:
            template = arguments.get("template") or ""
            lines = arguments.get("lines") or []
            brief = arguments.get("brief") or ""
            if not (
                isinstance(template, str) and isinstance(lines, list) and isinstance(brief, str)
            ):
                return ToolResult(
                    content=json.dumps(
                        {"error": "template and brief must be strings and lines a list"}
                    )
                )
            lines = [str(line) for line in lines][: meme.MAX_LINES]
            draw = arguments.get("draw")
            options = meme.MemeOptions(
                draw=draw.strip() if isinstance(draw, str) and draw.strip() else None,
            )
            self.log.info(
                "make_meme: template=%r brief=%r lines=%s", template, brief[:80], len(lines)
            )
            if not (template.strip() or brief.strip()):
                return ToolResult(content=json.dumps({"error": "Give a template or a brief."}))
            hosted, error = self._meme_or_infer(
                msg, template.strip() or brief.strip(), lines, options, named=bool(template.strip())
            )
            if hosted is None:
                return ToolResult(content=json.dumps({"error": error}))
            return ToolResult(content=json.dumps({"status": "ok", "message": hosted}))

        return [schema], {"make_meme": handler}

    def meme(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
        optlist: list,
        text: str,
    ) -> None:
        """[--font <name>] [--top] <template or image URL> | <caption> [| <caption> ...]  or  list [<word>]

        Captions one of about a thousand meme templates. Name the template
        (or paste an image URL), then the captions, separated by |. A blank
        caption leaves that box empty. --font changes the typeface (thick,
        thin, comic), --top puts the text in a white band above the picture.
        Or just say what you want: name no template (or no captions) and
        the bot picks one, writes the captions, and — when you describe who
        or what should be in the picture, or the template has a blank photo
        — edits the picture too.

        Examples:
          @meme drake | left on unread | left on read
          @meme distracted boyfriend | me | a new side project | my actual job
          @meme waiting for claude to finish
          @meme spirit halloween costume of my dad
          @meme distracted boyfriend where the boyfriend is a sysadmin and the girl in red is kubernetes
          @meme list cat
        """
        if self._is_old_message(msg):
            return
        if not self.registryValue("memeEnabled", msg.channel):
            irc.reply(_("Memes are turned off in this channel."))
            return

        options = meme.MemeOptions(
            font=next((v for opt, v in optlist if opt == "font"), None),
            layout_top=any(opt == "top" for opt, _v in optlist),
        )
        parsed = meme.parse_meme_request(text)
        if parsed is None:
            self._safe_error(irc, _("Usage: @meme <template> | <caption> | <caption>"))
            return
        template_query, lines = parsed

        head, _sep, word = template_query.partition(" ")
        if head.lower() == "list" and not lines:
            catalog = self._meme_catalog()
            if catalog is None:
                self._safe_error(irc, _("Meme templates are unavailable right now."))
                return
            picks = catalog.suggest(word, limit=10) if word.strip() else catalog.templates[:10]
            if word.strip():
                picks = [t for t in picks if meme.matches(t, word)] or picks
            if not picks:
                self._safe_reply(
                    irc,
                    _(
                        "No template matches '%s' (%d in all). Paste an image URL as the "
                        "template, or ask an operator for a memeAliases entry."
                    )
                    % (word.strip(), len(catalog)),
                )
                return
            listing = "; ".join(meme.describe_short(t) for t in picks)
            self._safe_reply(irc, _("Templates: %s — %d in all.") % (listing, len(catalog)))
            return

        pf = self._run_preflight(irc, msg, text, "meme", require_account=True)
        if pf.blocked:
            return

        # Typing goes up only for the picker path: a named template with
        # captions is a local render and nothing else. The grounding runs
        # there too — the same canon block and subject dossier @draw feeds
        # its planner, so "@meme spirit costume of <real person>" carries what
        # they look like into the edit.
        picks = not self._meme_names(template_query, has_lines=any(x.strip() for x in lines))
        stop_typing = self.llm_service._begin_typing(irc, msg) if picks else None
        # One permit for the whole picker path: dossier, meme_pick, the slot
        # pictures and the edit. Nothing inside may take another (no nesting).
        permit = self._llm_executor.permit() if picks else contextlib.nullcontext()
        try:
            with self._allow_concurrent(), permit:
                context = self._meme_grounding(pf, text) if picks else None
                hosted, error = self._meme_or_infer(
                    msg, template_query, lines, options, context=context
                )
        finally:
            if stop_typing:
                stop_typing()
        if hosted is None:
            self._safe_error(irc, error)
            return
        self._safe_reply(irc, hosted)

    def _meme_grounding(self, pf: PreflightResult, text: str) -> str | None:
        """Canon block + subject dossier for the picker, or None.

        The same two pre-stages @draw runs, in the same order (canon first;
        the dossier block says canon wins on overlap), minus the planner
        overlay wrapping: the picker takes the blocks as they are.

        Caller holds the executor permit (``meme`` takes it).
        """
        parts: list[str] = []
        verse_ctx = self._verse_context_for(pf, text)
        if verse_ctx:
            parts.append(verse_ctx)
        dossier_block = self._subject_dossier_for(
            pf.caller.identity.key, pf.caller.reply_target, text
        )
        if dossier_block:
            parts.append(dossier_block)
        return "\n\n".join(parts) or None

    meme = wrap(
        meme,
        [
            ("checkCapability", "llm.draw"),
            getopts(
                {
                    "font": "something",
                    "top": "",
                }
            ),
            "text",
        ],
    )

    # ------------------------------------------------------------------
    # @channels / @names — live LIST / NAMES lookups
    # ------------------------------------------------------------------

    def channels(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
        optlist: list,
        pattern: str | None,
    ) -> None:
        """[--min <n>] [<channel or glob>]

        Lists the busiest public channels on this network with user counts
        and topics. Give a channel name for that channel alone, or a glob
        like #linux* to filter. --min hides channels with fewer users.
        """
        if self._is_old_message(msg):
            return
        min_users = 0
        for opt, value in optlist:
            if opt == "min":
                min_users = int(value)
        rows = self._query_channels(irc)
        if rows is None:
            err = self._irc_queries.last_error(self._network_of(irc), "LIST")
            self._safe_error(irc, err or self._IRC_QUERY_SILENT)
            return
        self._safe_reply(irc, ircquery.format_channels(rows, pattern=pattern, min_users=min_users))

    channels = wrap(channels, [getopts({"min": "positiveInt"}), optional("something")])

    def model(self, irc: callbacks.Irc, msg: IrcMsg, args: list, channel: str) -> None:
        """[<channel>]

        Says which model answered the bot's last reply in the channel and how
        long ago, next to the chat model the channel is set to use. A runtime
        @config change can be undone by a restart, so the two can differ.
        """
        if self._is_old_message(msg):
            return
        configured = self.registryValue("assistantModel", channel) or "unset"
        served = (getattr(self, "_served_models", None) or {}).get(
            (self._network_of(irc), channel.lower())
        )
        if served is None:
            self._safe_reply(
                irc,
                _("No reply in %s since the bot started. Set to: %s.") % (channel, configured),
            )
            return
        name, at = served
        self._safe_reply(
            irc,
            _("Last reply in %s: %s, %s ago. Set to: %s.")
            % (channel, name, _format_ago(time.time() - at), configured),
        )

    model = wrap(model, ["channel"])

    def names(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
        channel: str | None,
    ) -> None:
        """[<channel>]

        Lists who is in <channel> at this moment, including channels the bot has
        not joined (the server hides secret and private channels itself).
        <channel> defaults to the current channel.
        """
        if self._is_old_message(msg):
            return
        target = channel or msg.channel
        if not target or not ircutils.isChannel(target):
            self._safe_error(irc, _("Give a channel name, like #chan."))
            return
        result = self._query_names(irc, target)
        if result is None:
            self._safe_error(irc, self._IRC_QUERY_SILENT)
            return
        if result.error is not None:
            self._safe_error(irc, f"{target}: {result.error}")
            return
        self._safe_reply(irc, ircquery.format_names(target, result.nicks))

    names = wrap(names, [optional("something")])

    def story(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
        text: str,
    ) -> None:
        """<brief>

        Generate an illustrated page from your description and post a link when
        it's ready. Either tells an illustrated tale OR explains a concept with
        diagrams as a learning aid — picks the mode from your brief. Draws a few
        AI illustrations, no verse mode required. Same image-spend gate as @draw
        (authenticated + llm.draw).

        Examples:
          @story an illustrated tale of stinky lads winning the pub quiz
          @story explain how photosynthesis works, with diagrams
        """
        # Skip ZNC playback messages
        if self._is_old_message(msg):
            return

        pf = self._run_preflight(irc, msg, text, "story", require_account=True)
        if pf.blocked:
            return
        caller = pf.caller
        channel = caller.reply_target

        brief = (text or "").strip()
        if not brief:
            self._safe_error(irc, _("Tell me what the story should be about."))
            return

        # Per-account cooldown (shared with verse_storybook), reserve-at-start.
        cooldown = int(self.registryValue("verseStorybookCooldownSeconds", channel) or 0)
        if self._storybook_cooldown_active(caller.identity.account, cooldown):
            self._safe_error(irc, _("Storybook cooldown is active — try again in a bit."))
            return

        # Fire-and-return: the page is rendered + posted asynchronously, so the
        # user gets the link when it's ready with no interim chatter (mirrors
        # the verse_storybook UX). Persona is the caller's avatar voice if set.
        # require_account above: the personal key is set.
        personal_key = caller.identity.personal_key
        persona = (self.db.get_avatar_persona(personal_key) or "") if personal_key else ""
        self._submit_storybook_job(caller=caller, persona=persona, brief=brief)

    story = wrap(story, [("checkCapability", "llm.draw"), "text"])

    def forget(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
        channel: str | None,
    ) -> None:
        """[<channel>]

        Clear volatile memory (conversation context) for the current or specified
        channel: your own thread AND the channel's shared recent-history, so stale
        bot answers stop feeding follow-ups. Volatile memory expires automatically
        after a timeout.
        """
        if self._is_old_message(msg):
            return
        caller = self._resolve_caller(irc, msg)
        origin = caller.reply_target
        # Default to current channel if not specified
        if channel is None:
            channel = origin
        self.context.clear(caller.identity.key, channel)
        # The channel-wide clear wipes the shared recent-history for EVERYONE in
        # the channel, so only allow it from within that same channel. Otherwise
        # any user could wipe an arbitrary channel's context from PM on demand
        # (griefing). A cross-channel / PM request still clears the caller's own
        # thread above; it just can't nuke a channel it isn't in.
        if ircutils.isChannel(channel) and channel == origin:
            # Stale bot text in the shared context poisons everyone's
            # follow-ups, and the shared window repopulates as people talk.
            self.context.clear_channel(channel)
        irc.reply(_("Context cleared."), prefixNick=False)

    forget = wrap(forget, [("checkCapability", "llm.ask"), optional("channel")])

    def memories(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
        text: str | None,
    ) -> None:
        """[<nick> | del(ete) <id> [<id>...] | edit <id> <text> | clear | cleanup [nick]]

        Manage your non-volatile memory (stored facts the bot remembers about you
        across conversations). Use 'delete <id> [<id>...]' to remove one or more
        memories, 'edit <id> <text>' to update one, 'clear' to remove all, or
        'cleanup' to trigger a cleanup pass. Bot owners can use 'memories <nick>'
        or 'memories cleanup <nick>' for other users.
        """
        if self._is_old_message(msg):
            return
        # Personal data: only an authenticated account has a personal key.
        caller = self._require_caller(irc, msg)
        key = caller.identity.personal_key if caller else None
        if caller is None or key is None:
            return

        if not text:
            # List own memories (newest first)
            self._memories_list(irc, key, key)
            return

        parts = text.split(None, 2)
        subcommand = parts[0].lower()

        if subcommand == "clear":
            count = self.db.delete_all_memories(key)
            label = "memory" if count == 1 else "memories"
            irc.reply(f"Cleared {count} {label}.", prefixNick=False)

        elif subcommand in ("delete", "del") and len(parts) >= 2:
            raw_ids = text.split()[1:]
            try:
                memory_ids = [int(x) for x in raw_ids]
            except ValueError:
                irc.error("Usage: memories delete <id> [<id> ...]")
                return
            deleted = sum(1 for mid in memory_ids if self.db.delete_memory(key, mid))
            if deleted == 0:
                irc.error("No matching memories found.")
            elif deleted == 1:
                irc.reply("Memory deleted.", prefixNick=False)
            else:
                irc.reply(f"Deleted {deleted} memories.", prefixNick=False)

        elif subcommand == "edit" and len(parts) == 3:
            try:
                memory_id = int(parts[1])
            except ValueError:
                irc.error("Usage: memories edit <id> <new text>")
                return
            new_text = parts[2].strip()
            if not new_text:
                irc.error("Usage: memories edit <id> <new text>")
                return
            if self.db.update_memory(key, memory_id, new_text):
                irc.reply("Memory updated.", prefixNick=False)
            else:
                irc.error("Memory not found or doesn't belong to you.")

        elif subcommand == "cleanup":
            # cleanup [nick] — nick requires owner
            if len(parts) >= 2:
                if not ircdb.checkCapability(msg.prefix, "owner"):
                    irc.error("Only bot owners can clean up other users' memories.")
                    return
                # Memories are keyed by the account, so resolve the typed nick
                # to its account or cleanup runs against the wrong key for any
                # user whose account differs from their nick.
                target = self._resolve_nick_to_identity(irc, parts[1])
            else:
                target = key
            channel = caller.reply_target
            if self._run_preflight(irc, msg, text, "ask", require_account=False).blocked:
                return
            # The permit lives here, not in _run_memory_cleanup: that also runs
            # as an assistant tool inside a turn that already holds one.
            with self._allow_concurrent(), self._llm_executor.permit():
                summary = self._run_memory_cleanup(target, channel)
            irc.reply(summary.message, prefixNick=False)

        elif len(parts) == 1:
            # Owner viewing another user's memories
            if not ircdb.checkCapability(msg.prefix, "owner"):
                irc.error("Usage: memories [del <id> | edit <id> <text> | clear | cleanup]")
                return
            # Query by resolved account (memories are account-keyed) but show
            # the operator the nick they typed.
            target = self._resolve_nick_to_identity(irc, parts[0])
            self._memories_list(irc, target, parts[0])

        else:
            irc.error("Usage: memories [del <id> | edit <id> <text> | clear | cleanup]")

    def _memories_list(self, irc: callbacks.Irc, nick: str, display_name: str) -> None:
        """List memories for a user using Limnoria's built-in pagination."""
        rows = self.db.get_memories(nick)
        if not rows:
            irc.reply(f"No memories stored for {display_name}.", prefixNick=False)
            return
        items = [f"[{r.id}] {r.fact}" for r in rows]
        irc.replies(items, joiner=" | ", prefixNick=False)

    memories = wrap(memories, [optional("text")])

    def instruct(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
        text: str | None,
    ) -> None:
        """[<instruction> | clear]

        Set persistent instructions that shape how @ask responds to you.
        Your instruction is prepended to the system prompt for every @ask call.

        Examples:
          @instruct You are Captain Picard. Respond in character.
          @instruct Respond only in haiku
          @instruct clear
          @instruct          (show current instruction)
        """
        if self._is_old_message(msg):
            return
        # Personal data: only an authenticated account has a personal key.
        caller = self._require_caller(irc, msg)
        key = caller.identity.personal_key if caller else None
        if caller is None or key is None:
            return

        if not text:
            current = self.db.get_instruction(key)
            if current:
                irc.reply(f"Current instruction: {current}", prefixNick=False)
            else:
                irc.reply("No instruction set. Use @instruct <text> to set one.", prefixNick=False)
            return

        is_clear = text.strip().lower() == "clear"

        if is_clear:
            if self.db.delete_instruction(key):
                irc.reply("Instruction cleared.", prefixNick=False)
            else:
                irc.reply("No instruction to clear.", prefixNick=False)
            return

        self.db.save_instruction(key, text)
        irc.reply("Instruction set.", prefixNick=False)

    instruct = wrap(instruct, [optional("text")])

    def tz(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
        text: str | None,
    ) -> None:
        """[<zone> | clear]

        Set your time zone so reminders like "tomorrow at 9am" fire at 9am
        your time. Use an IANA name (Area/City).

        Examples:
          @tz America/Toronto
          @tz clear
          @tz          (show current zone)
        """
        if self._is_old_message(msg):
            return
        # Personal data: only an authenticated account has a personal key.
        caller = self._require_caller(irc, msg)
        key = caller.identity.personal_key if caller else None
        if caller is None or key is None:
            return

        if not text:
            zone = parse_zone(self.db.get_user_timezone(key) or "")
            if zone is not None:
                label = UserTz(tz=zone, source=SOURCE_SET).label()
                irc.reply(f"Your time zone: {label}", prefixNick=False)
            else:
                irc.reply(
                    "No time zone set, so reminders use your client's clock or UTC. "
                    "Set one with @tz <Area/City>, e.g. @tz America/Toronto",
                    prefixNick=False,
                )
            return

        if text.strip().lower() == "clear":
            if self.db.delete_user_timezone(key):
                irc.reply("Time zone cleared.", prefixNick=False)
            else:
                irc.reply("No time zone to clear.", prefixNick=False)
            return

        zone = parse_zone(text)
        if zone is None:
            irc.reply(
                "Unknown time zone. Use an IANA name like America/Toronto or Europe/London.",
                prefixNick=False,
            )
            return

        self.db.save_user_timezone(key, zone.key)
        irc.reply(
            f"Time zone set: {UserTz(tz=zone, source=SOURCE_SET).label()}",
            prefixNick=False,
        )

    tz = wrap(tz, [optional("text")])

    def avatar(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
        text: str | None,
    ) -> None:
        """[<persona> | clear]

        Set the persona that shapes your avatar in verse-enabled channels.
        Independent of @instruct — this only affects the verse, not @ask.

        Examples:
          @avatar A moss-covered tree spirit who speaks in riddles.
          @avatar clear
          @avatar          (show current persona)
        """
        if self._is_old_message(msg):
            return
        # Personal data: only an authenticated account has a personal key.
        caller = self._require_caller(irc, msg)
        key = caller.identity.personal_key if caller else None
        if caller is None or key is None:
            return

        if not text:
            current = self.db.get_avatar_persona(key)
            if current:
                irc.reply(f"Current persona: {current}", prefixNick=False)
            else:
                irc.reply("No persona set. Use @avatar <text> to set one.", prefixNick=False)
            return

        channel = caller.channel
        is_clear = text.strip().lower() == "clear"
        new_summary = "" if is_clear else text

        # If in a verse-enabled channel with an active avatar, mirror to the
        # avatar's summary so /look and other display paths reflect the change
        # immediately. The user_avatar_personas table remains the source of
        # truth for the verse system prompt.
        if channel and self.registryValue("verseEnabled", channel):
            store = self._get_or_create_verse_store(channel)
            avatar_id = self._find_caller_avatar(store, key, caller.identity.raw_nick)
            if avatar_id is not None:
                entity = store.get_entity(avatar_id)
                if entity is not None and entity.status == "active":
                    # Through apply_direct like every other verse mutation so
                    # the change lands in the proposals audit trail.
                    store.apply_direct(
                        op="update_entity",
                        payload={"entity_id": avatar_id, "summary": new_summary},
                        source="avatar",
                        provenance="@avatar",
                    )

        if is_clear:
            if self.db.delete_avatar_persona(key):
                irc.reply("Persona cleared.", prefixNick=False)
            else:
                irc.reply("No persona to clear.", prefixNick=False)
            return

        self.db.save_avatar_persona(key, text)
        irc.reply("Persona set.", prefixNick=False)

    avatar = wrap(avatar, [optional("text")])

    def usage(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
        text: str | None,
    ) -> None:
        """[<nick or #channel>]

        Show API usage statistics.

        No argument in a channel: shows channel stats and your personal stats.
        No argument via PM: shows global overview (admin only).
        <nick>: shows that user's stats (scoped to current channel if in one).
        <#channel>: shows that channel's stats.
        """
        target = text.strip() if text else None

        # Strip IRC status prefixes (@op, +voice, %halfop) from nick targets
        if target and not ircutils.isChannel(target):
            target = target.lstrip("@+%")
        if target and ircutils.isChannel(target):
            self._usage_for_channel(irc, msg, target)
        elif target:
            self._usage_for_nick(irc, msg, target)
        elif msg.channel:
            self._usage_channel(irc, msg)
        else:
            if not ircdb.checkCapability(msg.prefix, "admin"):
                irc.error(_("You need the 'admin' capability to view global usage stats."))
                return
            self._usage_global(irc, msg)

    usage = wrap(usage, [optional("text")])

    def _usage_global(self, irc: callbacks.Irc, msg: IrcMsg) -> None:
        """Show global usage overview via PM (admin only)."""
        # Today: midnight UTC
        today_midnight = (
            datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        )

        # This month: first of month midnight UTC
        month_start = self._month_start_ts()

        today = self.db.get_usage_summary(since=today_midnight)
        month = self.db.get_usage_summary(since=month_start)
        top_users = self.db.get_usage_by_nick(since=month_start, limit=5)
        top_channels = self.db.get_usage_by_channel(since=month_start, limit=5)

        # Format response
        parts = []
        parts.append(f"Today: ${today.total_cost:.4f} ({today.total_requests} requests)")
        parts.append(f"This month: ${month.total_cost:.4f} ({month.total_requests} requests)")

        if top_users:
            # Show the nick, not the unauthenticated storage key's "!" prefix.
            user_parts = [
                f"{u.name.removeprefix(UNAUTHENTICATED_KEY_PREFIX)} ${u.total_cost:.4f}"
                for u in top_users
            ]
            parts.append(f"Top users: {', '.join(user_parts)}")

        if top_channels:
            chan_parts = [f"{c.name} ${c.total_cost:.4f}" for c in top_channels]
            parts.append(f"Top channels: {', '.join(chan_parts)}")

        # Global context stats
        ctx_global = self.context.get_stats()
        parts.append(
            f"Context: {ctx_global['active_conversations']} conversations,"
            f" {ctx_global['total_messages']} messages"
        )

        # LLM executor utilization (running/queued/max).
        ex = self._llm_executor
        parts.append(f"executor: {ex.running()}/{ex.queued()}/{ex.max_concurrency}")

        irc.reply(" | ".join(parts), private=True)

    def _usage_channel(self, irc: callbacks.Irc, msg: IrcMsg) -> None:
        """Show channel and personal usage stats in-channel."""
        channel = msg.channel
        identity = self._resolve_caller(irc, msg).identity

        # This month: first of month midnight UTC
        month_start = self._month_start_ts()

        chan_summary = self.db.get_usage_summary_for_channel(channel, since=month_start)
        nick_summary = self.db.get_usage_summary_for_nick(
            identity.key, since=month_start, channel=channel
        )
        chan_rank = self.db.get_channel_rank(channel, since=month_start)
        nick_rank = self.db.get_nick_rank(identity.key, since=month_start, channel=channel)

        # Format channel part
        chan_part = f"{channel} this month: ${chan_summary.total_cost:.4f}"
        chan_part += f" ({chan_summary.total_requests} requests"
        if chan_rank.rank > 0:
            chan_part += f", rank {chan_rank.rank}/{chan_rank.total} channels"
        chan_part += ")"

        # Format personal part
        nick_part = f"You: ${nick_summary.total_cost:.4f}"
        nick_part += f" ({nick_summary.total_requests} requests"
        if nick_rank.rank > 0:
            nick_part += f", rank {nick_rank.rank}/{nick_rank.total} users"
        nick_part += ")"

        # Format context part
        ctx_cfg = self._get_context_config(channel)
        ctx_stats = self.context.get_user_stats(identity.key, channel, config=ctx_cfg)
        if not ctx_stats["enabled"]:
            ctx_part = "Context: disabled"
        elif ctx_stats["message_count"] == 0:
            ctx_part = "Context: empty"
        else:
            remaining = ctx_stats["seconds_until_expiry"]
            minutes = remaining // 60
            ctx_part = f"Context: {ctx_stats['message_count']}/{ctx_stats['max_messages']} msgs"
            ctx_part += f", expires in {minutes}m" if minutes > 0 else ", expiring soon"

        irc.reply(f"{chan_part} | {nick_part} | {ctx_part}", prefixNick=False)

    def _usage_for_nick(self, irc: callbacks.Irc, msg: IrcMsg, nick: str) -> None:
        """Show usage stats for a specific nick.

        Resolves the target nick to a NickServ account before querying the
        database, so ``@usage OldNick`` finds stats logged under the account.
        The display still uses the nick the caller typed.
        """
        channel = msg.channel

        # Resolve target nick → account for the DB query
        identity = self._resolve_nick_to_identity(irc, nick)

        month_start = self._month_start_ts()

        nick_summary = self.db.get_usage_summary_for_nick(
            identity, since=month_start, channel=channel
        )
        nick_rank = self.db.get_nick_rank(identity, since=month_start, channel=channel)

        scope = f" in {channel}" if channel else ""
        nick_part = f"{nick}{scope} this month: ${nick_summary.total_cost:.4f}"
        nick_part += f" ({nick_summary.total_requests} requests"
        if nick_rank.rank > 0:
            nick_part += f", rank {nick_rank.rank}/{nick_rank.total} users"
        nick_part += ")"

        irc.reply(nick_part, prefixNick=False)

    def _usage_for_channel(self, irc: callbacks.Irc, msg: IrcMsg, channel: str) -> None:
        """Show usage stats for a specific channel."""
        month_start = self._month_start_ts()

        chan_summary = self.db.get_usage_summary_for_channel(channel, since=month_start)
        chan_rank = self.db.get_channel_rank(channel, since=month_start)

        chan_part = f"{channel} this month: ${chan_summary.total_cost:.4f}"
        chan_part += f" ({chan_summary.total_requests} requests"
        if chan_rank.rank > 0:
            chan_part += f", rank {chan_rank.rank}/{chan_rank.total} channels"
        chan_part += ")"

        irc.reply(chan_part, prefixNick=False)

    # Reminder helper methods (testable without Limnoria wrap decorator)

    def _pending_task_excludes(self, channel: str | None) -> frozenset[str]:
        """Tool names to drop from a user-facing request for ``channel``.

        When ``pendingTasksEnabled`` is off (the default) the five
        reminder/scheduled-task schemas and their prompt rules stay out of
        the request — they cost ~1,100 prompt tokens per completion. Only
        the chat entry route is gated: verse already excludes these tools,
        code/draw never see them, and reminder/scheduled fires must keep
        them so chains can list/cancel/reschedule.
        """
        if self.registryValue("pendingTasksEnabled", channel):
            return frozenset()
        return PENDING_TASK_TOOLS

    def _pending_task_fns(
        self,
        *,
        caller: Caller,
        irc: callbacks.Irc,
        msg: IrcMsg,
        pass_irc_msg_to_callbacks: bool = True,
    ) -> dict[str, Callable[..., object]]:
        """Build the unified pending-task tool dict for assistant calls.

        Returns callables for the consolidated ``list_pending_tasks`` /
        ``cancel_pending_task`` / ``cancel_all_pending_tasks`` /
        ``set_reminder`` / ``schedule_llm_task`` tool surface. The list and
        cancel paths span both reminders (``set_reminder``) and scheduled
        LLM tasks (``schedule_llm_task``) so the model can never look at
        only one kind when the user asks "what do I have scheduled?" or
        "cancel my X".

        ``pass_irc_msg_to_callbacks`` is False on the action-fire path: its
        ``synthetic_msg`` has no msgid, so passing ``irc``/``msg`` through
        would just invoke ``_react`` only to have its msgid check fail.
        """
        identity = caller.identity
        react_irc = irc if pass_irc_msg_to_callbacks else None
        react_msg = msg if pass_irc_msg_to_callbacks else None

        def set_reminder_fn(text: str) -> ToolCallbackResult:
            return self._remind_set_for_assistant(irc, msg, caller, text)

        def schedule_fn(
            *,
            when_natural: str,
            prompt: str,
            reply_target: str | None = None,
        ) -> dict[str, object]:
            # The action-fire path is not the user asking, so it does not probe;
            # neither does a relative time, which lands the same in any zone.
            probe = needs_zone(f"{when_natural} {prompt}")
            user_tz = self._resolve_user_tz(identity, irc=react_irc if probe else None)
            result = self.llm_service.schedule_llm_task(
                irc=irc,
                msg=msg,
                creator_nick=identity.raw_nick,
                account=identity.account,
                channel=caller.reply_target,
                when_natural=when_natural,
                prompt=prompt,
                reply_target=reply_target,
                user_tz=user_tz,
            )
            return {
                "status": result.status,
                "event_name": result.event_name,
                "fire_at": result.fire_at,
                "message": result.message,
                "note": result.note,
            }

        def list_pending_tasks_fn() -> list[dict[str, object]]:
            tasks: list[dict[str, object]] = []
            for name, data in self._get_user_reminders(identity):
                # event-name format: "llm_remind_<id>", stable across a
                # recurring chain's fires; the LLM only needs the trailing id.
                rid = name.split("_")[-1]
                tasks.append(
                    {
                        "kind": "reminder",
                        "id": rid,
                        "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(data.fire_at)),
                        "channel": data.channel,
                        "description": data.message,
                    }
                )
            for row in self.llm_service.list_scheduled_llm_tasks(
                creator_nick=identity.raw_nick, account=identity.account
            ):
                tasks.append(
                    {
                        "kind": "scheduled_task",
                        "id": row.event_name,
                        "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(row.fire_at)),
                        "channel": row.channel,
                        "description": row.prompt[:80],
                        "recurrence": (
                            f"every {row.recurrence_seconds}s"
                            if row.recurrence_seconds is not None
                            else row.recurrence_rrule
                        ),
                    }
                )
            return tasks

        def cancel_pending_task_fn(task_id: str) -> dict[str, object]:
            if task_id.startswith("llm_task_"):
                result = self.llm_service.cancel_scheduled_llm_task(
                    event_name=task_id,
                    creator_nick=identity.raw_nick,
                    account=identity.account,
                )
                return {
                    "status": result.status,
                    "kind": "scheduled_task",
                    "id": result.event_name,
                    "message": result.message,
                }
            result = self._remind_delete_for_assistant(
                identity, task_id, irc=react_irc, msg=react_msg
            )
            return {
                "status": "ok" if result.ok else "error",
                "kind": "reminder",
                "id": task_id,
                "message": result.message,
            }

        def cancel_all_pending_tasks_fn() -> dict[str, object]:
            reminder_msg = self._remind_clear_for_assistant(identity, irc=react_irc, msg=react_msg)
            scheduled_rows = self.llm_service.list_scheduled_llm_tasks(
                creator_nick=identity.raw_nick, account=identity.account
            )
            scheduled_cancelled = 0
            for row in scheduled_rows:
                result = self.llm_service.cancel_scheduled_llm_task(
                    event_name=row.event_name,
                    creator_nick=identity.raw_nick,
                    account=identity.account,
                )
                if result.status == "ok":
                    scheduled_cancelled += 1
            return {
                "reminders_message": reminder_msg,
                "scheduled_tasks_cancelled": scheduled_cancelled,
            }

        return {
            "set_reminder_fn": set_reminder_fn,
            "schedule_llm_task_fn": schedule_fn,
            "list_pending_tasks_fn": list_pending_tasks_fn,
            "cancel_pending_task_fn": cancel_pending_task_fn,
            "cancel_all_pending_tasks_fn": cancel_all_pending_tasks_fn,
        }

    def _get_user_reminders(self, caller: Identity) -> list[tuple[str, ReminderRow]]:
        """Get reminders belonging to a specific user.

        Match policy (:meth:`Identity.owns`): a reminder stored with an
        account belongs only to a caller authenticated as that account, so
        its owner keeps it across a nick change and a nick squatter never
        sees it; a reminder stored without one matches by nick.

        Args:
            caller: The requesting user's :class:`Identity`.

        Returns:
            List of ``(event_name, ReminderRow)`` pairs owned by ``caller``.
        """
        return [
            (row.event_name, row)
            for row in self.agenda.pending_reminders()
            if caller.owns(Identity(raw_nick=row.nick, account=row.account))
        ]

    def _format_reminders(
        self,
        reminders: list[tuple[str, ReminderRow]],
    ) -> str:
        """Format reminders list for display.

        Args:
            reminders: List of ``(event_name, ReminderRow)`` pairs.

        Returns:
            Formatted string for IRC display
        """
        parts = []
        for name, data in reminders:
            message = data.message
            action_prompt = data.action_prompt
            # Truncate long messages
            preview = message[:40] + "..." if len(message) > 40 else message
            # Extract ID from event name
            reminder_id = name.split("_")[-1]
            marker = " [auto]" if action_prompt else ""
            parts.append(f"#{reminder_id}: {preview}{marker}")
        return " | ".join(parts)

    def _find_user_reminder(self, caller: Identity, reminder_id: str) -> str | None:
        """Find a reminder event name by ID, scoped to the caller's identity.

        Args:
            caller: The requesting user's :class:`Identity`.
            reminder_id: Reminder ID (last part of event name).

        Returns:
            Event name if found and owned by the caller, ``None`` otherwise.
        """
        for name, _row in self._get_user_reminders(caller):
            if name.endswith(f"_{reminder_id}"):
                return name
        return None

    def _user_scheduled_tasks(self, caller: Identity) -> list[ScheduledLlmTaskRow]:
        """Scheduled LLM tasks owned by ``caller``.

        Ownership here is account-only, stricter than
        :meth:`_get_user_reminders`, which still matches nick-only reminders
        by nick: scheduling refuses without an
        authenticated account, so an accountless caller owns no tasks and
        gets an empty list. :meth:`_cancel_user_scheduled_task` applies the
        same rule, so the two halves of ``@remind`` agree on who owns what.
        """
        return self.llm_service.list_scheduled_llm_tasks(
            creator_nick=caller.raw_nick, account=caller.account
        )

    @staticmethod
    def _task_id(event_name: str) -> str:
        """The short id shown to users, from a ``llm_task_<id>`` event name."""
        return event_name.removeprefix("llm_task_")

    @classmethod
    def _format_scheduled_tasks(cls, rows: list[ScheduledLlmTaskRow]) -> str:
        """Format scheduled tasks for display, matching the reminder layout."""
        parts = []
        for row in rows:
            preview = row.prompt[:40] + "..." if len(row.prompt) > 40 else row.prompt
            parts.append(f"#{cls._task_id(row.event_name)}: {preview} [task]")
        return " | ".join(parts)

    def _cancel_user_scheduled_task(self, caller: Identity, task_id: str) -> bool:
        """Cancel one of ``caller``'s scheduled tasks. True when it went.

        Accepts either the short id shown by ``@remind list`` or the full
        ``llm_task_<id>`` event name. Ownership is re-checked by the service,
        so an id guessed off another user's task is refused there rather than
        here.
        """
        event_name = task_id if task_id.startswith("llm_task_") else f"llm_task_{task_id}"
        result = self.llm_service.cancel_scheduled_llm_task(
            event_name=event_name,
            creator_nick=caller.raw_nick,
            account=caller.account,
        )
        return result.status == "ok"

    @staticmethod
    def _describe_pending(reminders: int, tasks: int) -> str:
        """ "1 reminder and 2 scheduled tasks" — omitting whichever count is 0."""
        bits = []
        if reminders:
            bits.append(f"{reminders} reminder{'' if reminders == 1 else 's'}")
        if tasks:
            bits.append(f"{tasks} scheduled task{'' if tasks == 1 else 's'}")
        return " and ".join(bits)

    def _remind_list(self, irc: callbacks.Irc, caller: Identity) -> None:
        """List the calling user's pending reminders and scheduled tasks."""
        user_reminders = self._get_user_reminders(caller)
        tasks = self._user_scheduled_tasks(caller)
        if not user_reminders and not tasks:
            irc.reply(_("You have no pending reminders or scheduled tasks."))
            return
        parts = []
        if user_reminders:
            parts.append(self._format_reminders(user_reminders))
        if tasks:
            parts.append(self._format_scheduled_tasks(tasks))
        irc.reply(" | ".join(parts))

    def _get_reminders_for_target(self, target: str) -> list[tuple[str, ReminderRow]]:
        """Return reminders whose stored nick or account matches ``target``.

        Owner-only callers identify another user by either the nick that
        scheduled the reminder or the NickServ account it was stored
        under, case-insensitively.
        """
        target_lower = ircutils.toLower(target)
        return [
            (row.event_name, row)
            for row in self.agenda.pending_reminders()
            if ircutils.toLower(row.nick) == target_lower
            or (row.account and ircutils.toLower(row.account) == target_lower)
        ]

    def _find_reminder_for_target(self, target: str, reminder_id: str) -> str | None:
        """Find a single reminder owned by ``target`` by reminder ID."""
        for name, _row in self._get_reminders_for_target(target):
            if name.endswith(f"_{reminder_id}"):
                return name
        return None

    _REMINDER_MAX_SECONDS = 604800  # 7 days
    _REMINDER_MAX_PENDING_PER_USER = 25  # cap one-shot accumulation per user
    # A restatement of a reminder set this recently reads as a correction to it
    # ("no, wait, make it 2 minutes") rather than a second reminder. Keyed on
    # when the row was created, not on when it fires, so an amendment that
    # moves the time a long way is still recognised. Wide enough to span a
    # request and the follow-up that revises it, narrow enough that "take a
    # break in 1h" and, an hour later, "take a break in 3h" both stand.
    _REMINDER_AMEND_WINDOW_SECONDS = 120

    def _find_amendable_reminder(
        self, caller: Identity, *, message: str, now: float
    ) -> ReminderRow | None:
        """Return a just-created pending reminder of ``caller``'s restating ``message``.

        Only pending rows are compared. A fired one-shot has already been
        deleted, so asking for the same thing again afterwards still
        schedules a new one. A recurring reminder keeps the ``created_at`` of
        the request that set it across fires, so restating it later is a new
        reminder, not an amendment that ends the chain.
        """
        wanted = message.strip().casefold()
        for _name, row in self._get_user_reminders(caller):
            if (
                row.message.strip().casefold() == wanted
                and now - row.created_at <= self._REMINDER_AMEND_WINDOW_SECONDS
            ):
                return row
        return None

    def _react(self, irc: callbacks.Irc, msg: IrcMsg, emoji: str) -> bool:
        """Send a +draft/react reaction to ``msg``. Returns True if queued.

        No-ops cleanly when the server lacks message-tags or the incoming
        message has no msgid — caller should fall back to a text reply.
        """
        server_tags = getattr(msg, "server_tags", None) or {}
        msgid = server_tags.get("msgid")
        target = msg.args[0] if msg.args else ""
        if target and not ircutils.isChannel(target):
            # PM: react to the sender, not to the bot's own nick in args[0].
            target = getattr(msg, "nick", "") or target
        if not msgid or not target:
            self.log.info(
                "react_skipped emoji=%s reason=%s server_tag_keys=%s target=%r",
                emoji,
                "no_msgid" if not msgid else "no_target",
                sorted(server_tags.keys()) if server_tags else [],
                target,
            )
            return False
        return self.llm_service.send_reaction(irc, target, msgid, emoji)

    def _ack(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        emoji: str,
        fallback_text: str,
        *,
        prefixNick: bool = False,  # noqa: N803  (mirrors irc.reply kwarg)
    ) -> None:
        """React with `emoji`; fall back to text if the server can't carry it.

        `prefixNick` mirrors the kwarg on `irc.reply` — pass True when the call
        site previously called `irc.reply(text)` with the default prefix, False
        when it explicitly disabled prefixing.
        """
        if not self._react(irc, msg, emoji):
            irc.reply(fallback_text, prefixNick=prefixNick)

    def _cancel_reminder(self, event_name: str) -> None:
        """Cancel a reminder by deleting its row.

        The row is the only cancel truth: a fire already running finds it
        gone when it finishes and does not reschedule. Also takes back the ⏰
        put on the message that set it, so a cancelled reminder does not
        still look pending in the channel.
        """
        row = self.db.get_reminder(event_name)
        self.db.delete_reminder(event_name)
        irc = next(iter(world.ircs), None)
        if row is None or not row.reply_msgid or irc is None:
            return
        # A PM reminder stores a nick as its channel; the reaction went to
        # the sender, as in _react.
        target = Caller.from_row(row.nick, row.account, row.channel).reply_target
        self.llm_service.send_reaction(irc, target, row.reply_msgid, "⏰", remove=True)

    _CTCP_TIME_TIMEOUT_SECONDS = 3.0
    # A client clock goes stale when DST flips or its owner travels, so the
    # probe path re-asks after a week. Paths that cannot probe keep using the
    # old answer: last week's offset is closer than UTC.
    _CTCP_TIME_REFRESH_SECONDS = 7 * 86400

    def _resolve_user_tz(self, caller: Identity, *, irc: callbacks.Irc | None = None) -> UserTz:
        """The caller's zone: @tz, else their client clock, else UTC.

        ``irc`` enables the CTCP TIME probe. Leave it out on paths that must
        not block or talk to the user (a recurring reminder rescheduling
        itself); they still get the stored answer, however old.

        An accountless caller gets UTC: both stores key on account-or-nick,
        so their nick could be someone else's account.
        """
        if not caller.account:
            return UTC_DEFAULT
        zone = parse_zone(self.db.get_user_timezone(caller.key) or "")
        if zone is not None:
            return UserTz(tz=zone, source=SOURCE_SET)

        stored = self.db.get_clock_offset(caller.key)
        fresh = stored is not None and stored[1] + self._CTCP_TIME_REFRESH_SECONDS > time.time()
        if irc is not None and not fresh:
            offset = self._probe_ctcp_tz(irc, caller.raw_nick)
            seconds = None if offset is None else int(offset.utcoffset(None).total_seconds())
            self.db.save_clock_offset(caller.key, seconds)
        elif stored is not None and stored[0] is not None:
            offset = timezone(timedelta(seconds=stored[0]))
        else:
            offset = None
        if offset is not None:
            return UserTz(tz=offset, source=SOURCE_CTCP)
        return UTC_DEFAULT

    def _owner_tz(self, nick: str, account: str | None) -> tzinfo:
        """Zone for a stored reminder or task's owner, without probing."""
        return self._resolve_user_tz(Identity(raw_nick=nick, account=account)).tz

    def _probe_ctcp_tz(self, irc: callbacks.Irc, nick: str) -> timezone | None:
        """Ask ``nick``'s client for its clock and wait briefly for the answer.

        Many clients never answer (most mobile ones), and a bouncer answers
        with its own clock. The caller stores a silence like an answer, so
        the wait is paid at most once a week per user.
        """
        key = ircutils.toLower(nick)
        with self._ctcp_tz_lock:
            waiter = self._ctcp_tz_waiters.get(key)
            owner = waiter is None
            if owner:
                waiter = (threading.Event(), [])
                self._ctcp_tz_waiters[key] = waiter
        event, replies = waiter
        if owner:
            self._safe_queue(irc, ircmsgs.privmsg(nick, "\x01TIME\x01"))
        event.wait(self._CTCP_TIME_TIMEOUT_SECONDS)

        offset = offset_from_ctcp_time(replies[0], datetime.now(UTC)) if replies else None
        if owner:
            with self._ctcp_tz_lock:
                self._ctcp_tz_waiters.pop(key, None)
        log.info(
            "ctcp_time nick=%s answered=%s offset=%s",
            nick,
            bool(replies),
            offset,
        )
        return offset

    def _capture_ctcp_time(self, irc: callbacks.Irc, msg: IrcMsg) -> None:
        """Hand a CTCP TIME reply to the probe waiting on its sender.

        Unsolicited replies are ignored, so nobody can set another user's
        offset by sending the bot a crafted NOTICE.
        """
        if len(msg.args) < 2 or not ircutils.strEqual(msg.args[0], irc.nick):
            return
        text = msg.args[1]
        if not text.startswith("\x01TIME "):
            return
        with self._ctcp_tz_lock:
            waiter = self._ctcp_tz_waiters.get(ircutils.toLower(msg.nick or ""))
            if waiter is None or waiter[1]:
                return
            waiter[1].append(text[len("\x01TIME ") :].rstrip("\x01"))
        waiter[0].set()

    def _schedule_reminder(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        caller: Caller,
        text: str,
    ) -> ReminderScheduleResult:
        """Parse, validate, and schedule a reminder.

        The caller's raw nick is stored as the reminder's owning nick (used
        for the synthetic message prefix at fire time and as the fallback
        match when the caller has no account). The account is captured
        separately and is the preferred match key on lookup. The reply
        target is stored as the reminder's channel.

        A recurring reminder's later fires are the agenda's business (it
        moves this row in place and enforces the chain cap); this only ever
        starts a chain, at position 1.
        """
        identity = caller.identity
        channel = caller.reply_target
        # Resolved before taking an executor permit: the CTCP probe can wait
        # seconds on the user's client. A request with no wall clock in it
        # does not probe: "in 20 minutes" needs no zone.
        probe = needs_zone(text)
        user_tz = self._resolve_user_tz(identity, irc=irc if probe else None)

        with (
            self._trace_request("remind", identity.key, channel),
            self._allow_concurrent(),
            self._llm_executor.permit(),
        ):
            result = self.llm_service.parse_reminder(text, channel, user_tz=user_tz)

        if result.action == "clarify":
            # Not ok: a question read as success got a ⏰ and, in chat, had its
            # text dropped as a redundant ack, so the user never saw it.
            return ReminderScheduleResult(ok=False, message=result.confirmation, question=True)

        if result.seconds is None:
            return ReminderScheduleResult(
                ok=False, message="Could not determine when to set the reminder."
            )

        if result.seconds < 10:
            return ReminderScheduleResult(
                ok=False, message="Reminder must be at least 10 seconds from now."
            )

        if result.seconds > self._REMINDER_MAX_SECONDS:
            return ReminderScheduleResult(
                ok=False, message="Reminder can't be more than 7 days out."
            )

        now = time.time()
        pending = len(self._get_user_reminders(identity))
        if pending >= self._REMINDER_MAX_PENDING_PER_USER:
            return ReminderScheduleResult(
                ok=False,
                message=(
                    f"You already have {pending} pending reminders "
                    f"(cap {self._REMINDER_MAX_PENDING_PER_USER}). "
                    "Cancel some first."
                ),
            )

        reminder_message = result.message or text
        # msgid of the message that asked for the reminder, so the fire can be
        # threaded under it. Absent on servers without message-tags.
        reply_msgid = (getattr(msg, "server_tags", None) or {}).get("msgid") or ""
        # A second set_reminder for a request just handled is a restatement,
        # never a second reminder. It arrives two ways: the user revising
        # themselves ("no, wait, make it 2 minutes"), and the model re-issuing
        # the call because its own ack has not reached stored context yet — a
        # follow-up sent seconds later races the previous turn's write, so both
        # turns see identical history (observed 2026-09-03: one "in 60s"
        # request plus a follow-up produced two rows and two fires). Neither is
        # fixable at the history layer, so both collapse here: drop the earlier
        # row and let the new one take its place. A follow-up that carried no
        # new time re-parses to the same delay, so this degrades to "keep one
        # reminder" on its own.
        existing = self._find_amendable_reminder(identity, message=reminder_message, now=now)
        if existing is not None:
            self.log.info(
                "reminder_amended event=%s nick=%s old_fire_in=%is new_fire_in=%is",
                existing.event_name,
                identity.key,
                max(0, int(existing.fire_at - now)),
                int(result.seconds),
            )
            self._cancel_reminder(existing.event_name)

        event_name = f"llm_remind_{uuid.uuid4().hex[:12]}"
        try:
            self.db.save_reminder(
                event_name,
                identity.raw_nick,
                channel,
                reminder_message,
                now + result.seconds,
                action_prompt=result.action_prompt,
                account=identity.account,
                recurrence_seconds=result.recurrence_seconds,
                recurrence_rrule=result.recurrence_rrule,
                watch_mode=result.watch_mode,
                reply_msgid=reply_msgid,
            )
        except Exception as e:
            self.log.error("Failed to schedule reminder: %s", e)
            return ReminderScheduleResult(ok=False, message="Failed to set reminder.")
        self._arm_agenda()

        reply = self.llm_service.sanitize_output(result.confirmation)
        if result.note:
            reply = f"{reply} ({self.llm_service.sanitize_output(result.note)})"
        return ReminderScheduleResult(
            ok=True,
            message=reply,
            summary=f"{reminder_message} in {int(result.seconds)}s",
        )

    def _remind_set(self, irc: callbacks.Irc, msg: IrcMsg, caller: Caller, text: str) -> None:
        """Parse and schedule a natural language reminder via IRC."""
        result = self._schedule_reminder(irc, msg, caller, text)
        if result.ok:
            self._ack(irc, msg, "⏰", result.message, prefixNick=True)
            return
        if result.question:
            self._safe_reply(irc, result.message, prefixNick=True)
            return
        log.info(
            "remind_set blocked nick=%s reason=%s",
            caller.identity.key,
            result.message,
        )
        self._react(irc, msg, "❌")
        self._safe_error(irc, _(result.message))

    def _remind_set_for_assistant(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        caller: Caller,
        text: str,
    ) -> ToolCallbackResult:
        """Parse and schedule a reminder, returning a ToolCallbackResult for meta.

        Reacts ⏰ to ``msg`` on success and ❌ on cap/parse failure so the
        user gets a visual acknowledgment regardless of whether the model
        speaks. The chat reply path suppresses an empty post-tool reply
        via the structured ``last_successful_tool`` signal.
        """
        from .assistant import ToolCallbackResult as _ToolCallbackResult

        result = self._schedule_reminder(irc, msg, caller, text)
        if result.question:
            return _ToolCallbackResult(False, f"Not scheduled. Ask the user: {result.message}")
        self._react(irc, msg, "⏰" if result.ok else "❌")
        message = result.message
        if result.ok and result.summary:
            message = f"{message} (now pending: {result.summary})"
        return _ToolCallbackResult(result.ok, message)

    def _remind_delete_for_assistant(
        self,
        caller: Identity,
        reminder_id: str,
        *,
        irc: callbacks.Irc | None = None,
        msg: IrcMsg | None = None,
    ) -> ToolCallbackResult:
        """Delete a reminder by ID, scoped to the caller's identity.

        When ``irc``/``msg`` are provided, reacts 👍 on success and ❌
        when no matching reminder exists (so the chat path can suppress
        the empty post-tool reply without leaving the user wondering).
        """
        from .assistant import ToolCallbackResult as _ToolCallbackResult

        target = self._find_user_reminder(caller, reminder_id)
        if target is None:
            if irc is not None and msg is not None:
                self._react(irc, msg, "❌")
            return _ToolCallbackResult(False, f"Reminder {reminder_id} not found.")

        self._cancel_reminder(target)
        if irc is not None and msg is not None:
            self._react(irc, msg, "👍")
        return _ToolCallbackResult(True, f"Deleted reminder {reminder_id}.")

    def _remind_clear_for_assistant(
        self,
        caller: Identity,
        *,
        irc: callbacks.Irc | None = None,
        msg: IrcMsg | None = None,
    ) -> str:
        """Cancel all pending reminders for the caller in one shot.

        Snapshots the user's reminders then removes each — single atomic
        operation from the LLM's perspective so a recurring reminder can't
        slip in a fire between the model's tool calls.
        """
        user_reminders = self._get_user_reminders(caller)
        for name, _data in user_reminders:
            self._cancel_reminder(name)
        count = len(user_reminders)
        if irc is not None and msg is not None:
            self._react(irc, msg, "👍" if count else "👌")
        if count == 0:
            return "No pending reminders to cancel."
        if count == 1:
            return "Cancelled 1 reminder."
        return f"Cancelled {count} reminders."

    def remind(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
        text: str | None,
    ) -> None:
        """[<reminder text> | list | del(ete) <id> [<id>...] | clear]

        Set and manage reminders using natural language. If your reminder
        asks the bot to *do* something (look something up, check a status,
        fetch a URL), it will run that as an LLM query at fire time;
        otherwise it just echoes your text. Reminders marked [auto] in
        `list` are LLM actions.

        `list`, `delete` and `clear` also cover your scheduled tasks, which
        are marked [task]; delete one by the id `list` shows.

        Examples:
          @remind in 30 minutes check the build
          @remind in 2 hours check status of CVE-2026-31431 in Debian
          @remind list
          @remind delete abc1
          @remind clear
          @remind admin list <nick>      (owner only)
          @remind admin del <nick> <id>  (owner only)
          @remind admin clear <nick>     (owner only)
        """
        if self._is_old_message(msg):
            return
        caller = self._resolve_caller(irc, msg)
        identity = caller.identity

        if not text:
            self._remind_list(irc, identity)
            return

        parts = text.split(None, 1)
        subcommand = parts[0].lower()

        if subcommand == "admin":
            if not ircdb.checkCapability(msg.prefix, "owner"):
                irc.error(_("Only bot owners can manage other users' reminders."))
                return
            self._remind_admin(irc, msg, parts[1] if len(parts) >= 2 else "")
            return

        if subcommand == "list":
            self._remind_list(irc, identity)

        elif subcommand in ("delete", "del") and len(parts) >= 2:
            raw_ids = text.split()[1:]
            reminders_deleted = 0
            tasks_deleted = 0
            for rid in raw_ids:
                target = self._find_user_reminder(identity, rid)
                if target:
                    self._cancel_reminder(target)
                    reminders_deleted += 1
                elif self._cancel_user_scheduled_task(identity, rid):
                    tasks_deleted += 1
            if reminders_deleted == 0 and tasks_deleted == 0:
                self._react(irc, msg, "❌")
                irc.error(_("No matching reminders or scheduled tasks found."))
            else:
                what = self._describe_pending(reminders_deleted, tasks_deleted)
                self._ack(irc, msg, "👍", f"Cancelled {what}.")

        elif subcommand == "clear":
            user_reminders = self._get_user_reminders(identity)
            tasks = self._user_scheduled_tasks(identity)
            if not user_reminders and not tasks:
                self._ack(irc, msg, "👌", _("No reminders or scheduled tasks to clear."))
                return
            for name, _data in user_reminders:
                self._cancel_reminder(name)
            tasks_cleared = sum(
                1 for row in tasks if self._cancel_user_scheduled_task(identity, row.event_name)
            )
            what = self._describe_pending(len(user_reminders), tasks_cleared)
            if not what:
                # Everything listed fired or was cancelled underneath us.
                self._ack(irc, msg, "👌", _("Nothing left to clear."))
                return
            self._ack(irc, msg, "👍", f"Cleared {what}.")

        else:
            self._remind_set(irc, msg, caller, text)

    def _cancel_scheduled_llm_task_admin(self, event_name: str) -> None:
        """Owner-only cancel: skip the ownership check and delete the row."""
        self.db.delete_scheduled_llm_task(event_name)

    def _remind_admin(self, irc: callbacks.Irc, msg: IrcMsg, rest: str) -> None:
        """Owner-only dispatcher for cross-user reminder management.

        Subcommands: ``list <target>``, ``del <target> <id> [<id>...]``,
        ``clear <target>``.  ``target`` matches stored nick or account
        case-insensitively.  ``clear`` cancels both ``reminders`` and
        ``scheduled_llm_tasks`` rows because users speak of both as
        "reminders" and the difference is invisible to them.
        """
        tokens = rest.split()
        if len(tokens) < 2:
            irc.error(_("Usage: remind admin <list|del|clear> <nick> [<id>...]"))
            return

        action = tokens[0].lower()
        target = tokens[1]

        if action == "list":
            reminder_rows = self._get_reminders_for_target(target)
            task_rows = self.db.load_scheduled_llm_tasks_for_target(target)
            if not reminder_rows and not task_rows:
                irc.reply(f"No pending reminders or scheduled tasks for {target}.")
                return
            parts = []
            if reminder_rows:
                parts.append(self._format_reminders(reminder_rows))
            for row in task_rows:
                preview = (row.prompt[:40] + "...") if len(row.prompt) > 40 else row.prompt
                parts.append(f"task:{row.event_name}: {preview}")
            irc.reply(f"{target}: " + " | ".join(parts))

        elif action in ("delete", "del"):
            if len(tokens) < 3:
                irc.error(_("Usage: remind admin del <nick> <id> [<id>...]"))
                return
            deleted = 0
            for rid in tokens[2:]:
                reminder_event = self._find_reminder_for_target(target, rid)
                if reminder_event:
                    self._cancel_reminder(reminder_event)
                    deleted += 1
                    continue
                # Accept the short id `@remind list` shows as well as the full
                # event name printed by `admin list`.
                event_name = rid if rid.startswith("llm_task_") else f"llm_task_{rid}"
                task_row = self.db.get_scheduled_llm_task(event_name)
                if task_row is not None and (
                    ircutils.toLower(task_row.creator_nick) == ircutils.toLower(target)
                    or (
                        task_row.account
                        and ircutils.toLower(task_row.account) == ircutils.toLower(target)
                    )
                ):
                    self._cancel_scheduled_llm_task_admin(event_name)
                    deleted += 1
            if deleted == 0:
                self._react(irc, msg, "❌")
                irc.error(_("No matching reminders or tasks found."))
            else:
                label = "entry" if deleted == 1 else "entries"
                self._ack(irc, msg, "👍", f"Cancelled {deleted} {label} for {target}.")

        elif action == "clear":
            reminder_rows = self._get_reminders_for_target(target)
            task_rows = self.db.load_scheduled_llm_tasks_for_target(target)
            total = len(reminder_rows) + len(task_rows)
            if total == 0:
                self._ack(irc, msg, "👌", f"Nothing to clear for {target}.")
                return
            for name, _data in reminder_rows:
                self._cancel_reminder(name)
            for row in task_rows:
                self._cancel_scheduled_llm_task_admin(row.event_name)
            label = "entry" if total == 1 else "entries"
            self._ack(irc, msg, "👍", f"Cleared {total} {label} for {target}.")

        else:
            irc.error(_("Usage: remind admin <list|del|clear> <nick> [<id>...]"))

    # Gated like its siblings (@ask, @forget). @remind reaches persisted,
    # account-owned rows — reminders and, since the list/cancel merge, scheduled
    # tasks — so it should not be the one command anyone can reach ungated.
    # llm.ask rather than a bucket of its own: action reminders already bill to
    # the ask rate limit at fire time.
    remind = wrap(remind, [("checkCapability", "llm.ask"), optional("text")])

    # =========================================================================
    # Verse subsystem
    # =========================================================================

    # -------------------------------------------------------------------------
    # Verse subsystem channel helpers
    # -------------------------------------------------------------------------

    def _all_known_channels(self) -> set[str]:
        """Channels seen on any connected Irc. Used by ``_verse_enabled_channels``."""
        seen: set[str] = set()
        for irc_conn in world.ircs:
            seen.update(irc_conn.state.channels.keys())
        return seen

    def _verse_enabled_channels(self) -> list[str]:
        """All channels with verseEnabled=True. Read from registry every call
        (callers are not in hot paths)."""
        out: list[str] = []
        for ch in self._all_known_channels():
            if self.registryValue("verseEnabled", ch):
                out.append(ch)
        return out

    # -------------------------------------------------------------------------
    # Daily verse compaction timer (PR 3 / E3)
    # -------------------------------------------------------------------------

    def _register_compaction_timer(self) -> None:
        """Arm the next single-shot compaction firing.

        Idempotent: ``register_daily_timer`` cancels any existing event
        with the same name first, so callers can re-arm without bothering
        to cancel.
        """
        from llm.verse.compaction import register_daily_timer

        try:
            fire_at = self.registryValue("verseCompactionDailyAt") or "03:00"
        except Exception:
            # Registry key not yet defined (F1 adds it). Fall back so
            # the timer still arms on un-configured installs.
            fire_at = "03:00"
        try:
            register_daily_timer(
                schedule_module=schedule,
                fire_at_local=fire_at,
                callback=self._compaction_tick,
                name=self._compaction_timer_name,
            )
        except Exception:
            # On addEvent failure the timer is gone until next plugin
            # reload — daily cadence makes this acceptable.
            self.log.exception("verse: failed to register compaction timer")

    def _cancel_compaction_timer(self) -> None:
        from llm.verse.compaction import cancel_daily_timer

        cancel_daily_timer(schedule_module=schedule, name=self._compaction_timer_name)

    def _compaction_tick(self) -> None:
        """One firing of the daily timer: offload the pass, then re-arm.

        ``_run_compaction_pass`` makes one blocking LLM call per
        verse-enabled channel. This callback fires on Limnoria's scheduler
        thread — the IRC driver's main loop — which cannot flush the
        outbound queue or answer PINGs while a callback runs. So the work is
        handed to ``_llm_executor`` instead of running inline, and
        ``_dispatch_addressed_async``
        carries the full rationale for why blocking the driver thread is the
        bug to avoid. The re-arm runs in ``finally`` so a failed submit never
        kills the timer — the next day still gets a shot.
        """
        try:
            if not self._llm_executor.closing:
                self._llm_executor.submit("verse_compaction", self._run_compaction_pass)
        except Exception:
            self.log.exception("verse compaction submit failed")
        finally:
            self._register_compaction_timer()

    def _compaction_settings(self):
        """Shared config reads for the daily pass and @versecompact.

        Returns (min_keep, model, client). NB: explicit ``int(value)`` rather
        than ``int(value or 20)`` — 0 is a legitimate registry value
        (NonNegativeInteger accepts it) and ``or`` would coerce it to 20.
        """
        from llm.verse import compaction as _compaction

        try:
            _raw_min_keep = self.registryValue("verseCompactionMinKeepEvents")
        except Exception:
            _raw_min_keep = 20
        try:
            min_keep = int(_raw_min_keep) if _raw_min_keep is not None else 20
        except (TypeError, ValueError):
            min_keep = 20
        model = self.registryValue("verseCompactionModel") or "anthropic/claude-haiku-5-5"
        client = _compaction.ProviderVerseClient()
        return min_keep, model, client

    def _channel_retention_days(self, channel: str, default: int = 30) -> int:
        """Per-channel verseEventRetentionDays, honouring zero (=disabled)."""
        try:
            raw = self.registryValue("verseEventRetentionDays", channel)
        except Exception:
            raw = default
        try:
            return int(raw) if raw is not None else default
        except (TypeError, ValueError):
            return default

    def _log_compaction_usage(self, *, op: str, model: str, usage, channel: str) -> None:
        # Renamed from the deleted loom subsystem's "loom"/"loom:" identity;
        # historical rows were re-keyed in the same deploy so @usage stays
        # coherent (UPDATE usage SET nick, command WHERE nick='loom').
        self.db.log_usage(
            nick="verse_compaction",
            channel=channel,
            command=f"compaction:{op}",
            model=model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cost=usage.cost,
        )

    def _compact_channel(
        self, channel: str, *, min_keep: int, model: str, client, run_aging: bool
    ) -> tuple[CompactionOutcome, AgingOutcome | None]:
        """Aging (optional) + retention compaction for ONE channel.

        Aging failures are logged and swallowed (independent of compaction);
        compact_verse exceptions propagate for the caller to handle.
        """
        from llm.verse import compaction as _compaction
        from llm.verse.aging import age_auto_created_entities

        store = self._get_or_create_verse_store(channel)
        retention_days = self._channel_retention_days(channel)
        aging_outcome: AgingOutcome | None = None
        if run_aging:
            # Aging runs BEFORE compaction. compact_verse's digest heartbeat
            # bumps last_seen_ts=now() on every entity in the new digest, so
            # running it first would refresh long-silent NPCs and defeat
            # verseAutoEntityRetireDays.
            try:
                retire_days = self.registryValue("verseAutoEntityRetireDays", channel)
                aging_outcome = age_auto_created_entities(
                    store,
                    retire_after_days=retire_days,
                    now=time.time,
                )
            except Exception:
                self.log.exception("verse aging failed for %s; continuing", channel)
        outcome = _compaction.compact_verse(
            store,
            retention_days=retention_days,
            min_keep_events=min_keep,
            model=model,
            client=client,
            log_usage=lambda *, op, model, usage: self._log_compaction_usage(
                op=op, model=model, usage=usage, channel=channel
            ),
            now=time.time,
        )
        return outcome, aging_outcome

    def _run_compaction_pass(self) -> None:
        """Walk every verse-enabled channel and compact it.

        Per-channel failures are logged and swallowed so one bad verse
        doesn't abort the rest of the pass.
        """
        min_keep, model, client = self._compaction_settings()
        for channel in self._verse_enabled_channels():
            # Defensive re-check — _verse_enabled_channels already filters,
            # but the registry could flip mid-pass.
            try:
                if not self.registryValue("verseEnabled", channel):
                    continue
            except Exception:
                continue
            try:
                outcome, aging_outcome = self._compact_channel(
                    channel, min_keep=min_keep, model=model, client=client, run_aging=True
                )
            except Exception:
                self.log.exception("verse compaction failed for %s; continuing", channel)
                continue
            msg = _format_compaction_outcome(outcome, aging_outcome, min_keep_events=min_keep)
            self.log.info("compaction outcome for %s: %s", channel, msg)

    def _get_or_create_verse_store(self, channel: str) -> VerseStore:
        """Return the VerseStore for *channel*, creating it lazily on first access.

        Thread-safe: guarded by ``_verse_stores_lock``.  The store itself is
        safe to use outside the lock — it carries its own write lock.
        """
        key = ircutils.toLower(channel)
        with self._verse_stores_lock:
            store = self._verse_stores.get(key)
            if store is None:
                store = VerseStore(self._verse_data_dir(), channel)
                self._verse_stores[key] = store
            return store

    def _verse_data_dir(self) -> Path:
        return Path(conf.supybot.directories.data()) / "verse"

    def verseopt(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
        mode: str,
    ) -> None:
        """<in|out>

        Opt your avatar in or out of the verse for this channel.

          @verseopt in   — create or reactivate your avatar; shows the opening scene.
          @verseopt out  — retire your avatar until you opt back in.

        Requires the llm.verse capability and a verse-enabled channel.
        """
        if self._is_old_message(msg):
            return
        channel = msg.args[0] if msg.args else None
        if not channel or not ircutils.isChannel(channel):
            irc.error(_("This command must be used in a channel."), prefixNick=False)
            return

        if not self.registryValue("verseEnabled", channel):
            irc.reply(self._NO_VERSE_REPLY, prefixNick=False)
            return

        identity = self._resolve_caller(irc, msg).identity
        nick = identity.raw_nick
        account = identity.account
        personal_key = identity.personal_key

        if mode == "in":
            persona_text = (self.db.get_avatar_persona(personal_key) or "") if personal_key else ""
            store = self._get_or_create_verse_store(channel)
            result = store.opt_in_avatar(nick, account, persona_text)
            reply = result.scene_text
            if result.was_already_opted_in:
                reply = "You are already opted in. " + reply
            irc.reply(reply, prefixNick=False)

        else:  # mode == "out"
            store = self._get_or_create_verse_store(channel)
            entity_id = self._find_caller_avatar(store, account, nick)
            if entity_id is None:
                irc.reply("You don't have an avatar in this channel.", prefixNick=False)
                return
            store.unlink_avatar(entity_id)
            irc.reply(
                "Avatar retired. Use @verseopt in to rejoin.",
                prefixNick=False,
            )

    verseopt = wrap(verseopt, [("checkCapability", "llm.verse"), ("literal", ("in", "out"))])

    # ------------------------------------------------------------------
    # Helpers shared by @verse / @look / @who
    # ------------------------------------------------------------------

    _NO_VERSE_REPLY = "This channel doesn't have a verse. Ask the operator to set verseEnabled."
    _NO_AVATAR_REPLY = "You don't have an avatar in this channel. Use @verseopt in to join."

    def _check_verse_channel(self, irc: callbacks.Irc, msg: IrcMsg) -> str | None:
        """Return the channel name if verse is enabled, else reply and return None."""
        channel = msg.args[0] if msg.args else None
        if not channel or not ircutils.isChannel(channel):
            irc.error(_("This command must be used in a channel."), prefixNick=False)
            return None
        if not self.registryValue("verseEnabled", channel):
            irc.reply(self._NO_VERSE_REPLY, prefixNick=False)
            return None
        return channel

    @staticmethod
    def _avatar_place(store: VerseStore, entity_id: int):
        """Return the active place the avatar is at, or None. The 'location'
        attribute holds the place NAME (opt_in_avatar / verse_move)."""
        location = store.get_attribute(entity_id, "location")
        if location is None:
            return None
        return store.find_entity_by_name(location, kind="place", active_only=True)

    def _avatar_scene_oneliner(self, store: VerseStore, entity_id: int) -> str:
        """Build an IRC-friendly one-liner for the avatar's current location."""
        place = self._avatar_place(store, entity_id)
        if place is None:
            return "You are nowhere in particular."
        return f"You are at {place.name}. {place.summary}".rstrip()

    # ------------------------------------------------------------------
    # @verse — show caller's current scene one-liner
    # ------------------------------------------------------------------

    def verse(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
    ) -> None:
        """(takes no arguments)

        Show your current scene one-liner in the verse.

          @verse — display where your avatar currently is.

        Requires the llm.verse capability and a verse-enabled channel.
        """
        channel = self._check_verse_channel(irc, msg)
        if channel is None:
            return

        identity = self._resolve_caller(irc, msg).identity
        store = self._get_or_create_verse_store(channel)
        entity_id = self._find_caller_avatar(store, identity.account, identity.raw_nick)
        if entity_id is None:
            irc.reply(self._NO_AVATAR_REPLY, prefixNick=False)
            return

        irc.reply(self._avatar_scene_oneliner(store, entity_id), prefixNick=False)

    verse = wrap(verse, [("checkCapability", "llm.verse")])

    # ------------------------------------------------------------------
    # @look [target] — scene or entity description
    # ------------------------------------------------------------------

    def look(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
        target: str | None = None,
    ) -> None:
        """[<target>]

        Show your current scene, or describe a named entity in the verse.

          @look          — show where you are (same as @verse).
          @look <name>   — describe an entity by name.

        Requires the llm.verse capability and a verse-enabled channel.
        """
        channel = self._check_verse_channel(irc, msg)
        if channel is None:
            return

        store = self._get_or_create_verse_store(channel)

        if target is None:
            # No target: show caller's scene (avatar required).
            identity = self._resolve_caller(irc, msg).identity
            entity_id = self._find_caller_avatar(store, identity.account, identity.raw_nick)
            if entity_id is None:
                irc.reply(self._NO_AVATAR_REPLY, prefixNick=False)
                return
            irc.reply(self._avatar_scene_oneliner(store, entity_id), prefixNick=False)
        else:
            # Target given: look up entity by name or alias (same resolution
            # as @canon, so "@look Archie" works on an alias too).
            entity = store.find_entity_by_name_or_alias(target)
            if entity is None:
                irc.reply("Nothing matches.", prefixNick=False)
                return
            irc.reply(f"{entity.name}: {entity.summary}", prefixNick=False)

    look = wrap(look, [("checkCapability", "llm.verse"), optional("text")])

    # ------------------------------------------------------------------
    # @who — list active avatars and their locations
    # ------------------------------------------------------------------

    def who(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
    ) -> None:
        """(takes no arguments)

        List active avatars and their current locations in the verse.

          @who — roster of opted-in avatars.

        Requires the llm.verse capability and a verse-enabled channel.
        """
        channel = self._check_verse_channel(irc, msg)
        if channel is None:
            return

        store = self._get_or_create_verse_store(channel)
        avatars = store.list_entities_by_kind("avatar", status="active")
        if not avatars:
            irc.reply("Nobody is opted in here yet.", prefixNick=False)
            return

        parts: list[str] = []
        for avatar in avatars:
            place = self._avatar_place(store, avatar.id)
            parts.append(f"{avatar.name} (at {place.name})" if place else avatar.name)

        irc.reply(", ".join(parts), prefixNick=False)

    who = wrap(who, [("checkCapability", "llm.verse")])

    # ------------------------------------------------------------------
    # @canon lock|unlock|forget <name> — author-gated durable canon
    # ------------------------------------------------------------------

    def canon(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
        action: str,
        name: str,
    ) -> None:
        """<lock|unlock|forget> <name>

        Lock or release a character as durable canon (always remembered,
        aging-exempt). 'forget' is an alias for 'unlock'.

          @canon lock <name>   — mark a character as durable canon.
          @canon unlock <name> — release it.
          @canon forget <name> — same as unlock.

        Requires the llm.verse.edit capability and a verse-enabled channel.
        """
        channel = self._check_verse_channel(irc, msg)
        if channel is None:
            return
        store = self._get_or_create_verse_store(channel)
        ent = store.find_entity_by_name_or_alias(name)
        if ent is None:
            irc.error(f"No such character: {name}", prefixNick=False)
            return
        store.set_author_locked(ent.id, action == "lock")
        irc.replySuccess()

    canon = wrap(
        canon,
        [
            ("checkCapability", "llm.verse.edit"),
            ("literal", ("lock", "unlock", "forget")),
            "text",
        ],
    )

    # ------------------------------------------------------------------
    # @versedump #chan [--format=json]  — owner dump of full verse state
    # ------------------------------------------------------------------

    def versedump(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
        text: str | None = None,
    ) -> None:
        """[#channel] [--format=json]

        Dump the full verse state for the given channel as JSON.

          @versedump #chan           — dump in JSON (default).
          @versedump #chan --format=json  — explicit JSON.

        YAML is not supported (pyyaml is not a project dependency).
        Requires the llm.verse.gm capability.
        """
        # Parse channel and optional --format from the free-text args.
        raw = (text or "").split()
        channel: str | None = None
        fmt = "json"
        for token in raw:
            if token.startswith("--format="):
                fmt = token[len("--format=") :]
            elif ircutils.isChannel(token):
                channel = token

        if channel is None:
            # Fall back to the channel the command arrived in.
            ch = msg.args[0] if msg.args else None
            if ch and ircutils.isChannel(ch):
                channel = ch
            else:
                irc.error("Specify a channel: @versedump #chan", prefixNick=False)
                return

        if fmt not in ("json",):
            irc.reply(
                "Unsupported format. Only --format=json is supported.",
                prefixNick=False,
            )
            return

        store = self._get_or_create_verse_store(channel)

        # Collect all entities (all kinds, all statuses) with their attributes.
        with store.read_connection() as conn:
            entity_rows = conn.execute(
                "SELECT id, kind, name, summary, status, created_at, updated_at"
                " FROM entities ORDER BY id ASC"
            ).fetchall()
            avatar_link_rows = conn.execute(
                "SELECT entity_id, nick, account FROM avatar_link ORDER BY entity_id ASC"
            ).fetchall()
            alias_rows = conn.execute(
                "SELECT entity_id, alias FROM entity_alias ORDER BY entity_id ASC, alias ASC"
            ).fetchall()

        entities_out = []
        for row in entity_rows:
            eid = row[0]
            entities_out.append(
                {
                    "id": eid,
                    "kind": row[1],
                    "name": row[2],
                    "summary": row[3],
                    "status": row[4],
                    "created_at": row[5],
                    "updated_at": row[6],
                    "attributes": store.list_attributes(eid),
                }
            )

        # Relations (all, no filter).
        relations_raw = store.list_relations()
        relations_out = [
            {"id": r.id, "from_id": r.from_id, "to_id": r.to_id, "kind": r.kind, "note": r.note}
            for r in relations_raw
        ]

        # Recent events (capped at 200 to keep IRC PMs manageable).
        events_raw = store.recent_events(limit=200)
        events_out = [
            {
                "id": ev.id,
                "ts": ev.ts,
                "summary": ev.summary,
                "entity_ids": list(ev.entity_ids),
                "source": ev.source,
            }
            for ev in events_raw
        ]

        avatar_links_out = [
            {"entity_id": row[0], "nick": row[1], "account": row[2]} for row in avatar_link_rows
        ]
        aliases_out = [{"entity_id": row[0], "alias": row[1]} for row in alias_rows]

        # Proposals are apply_direct's audit trail — export them so the dump
        # is a faithful backup instead of a hardcoded empty list.
        proposals_out = [
            {
                "id": pr.id,
                "created_at": pr.created_at,
                "op": pr.op,
                "payload": pr.payload,
                "provenance": pr.provenance,
                "status": pr.status,
            }
            for pr in store.list_proposals(limit=1000)
        ]

        dump = {
            "schema_version": 2,
            "channel": channel,
            "entities": entities_out,
            "relations": relations_out,
            "events": events_out,
            "avatar_links": avatar_links_out,
            "aliases": aliases_out,
            "proposals": proposals_out,
        }

        # The dump is a single fat JSON line — for an active verse it can
        # easily exceed Limnoria's pagination. Publish to the bot's HTTP
        # pastebin and reply with just the URL; fall back to inline only
        # if the pastebin write fails (e.g. http server not configured).
        body = json.dumps(dump, indent=2)
        markdown_body = f"# versedump {channel}\n\n```json\n{body}\n```\n"
        url = self.llm_service.save_markdown_to_http(markdown_body)
        if url:
            irc.reply(
                f"versedump {channel}: {url}",
                prefixNick=False,
            )
        else:
            irc.reply(json.dumps(dump, separators=(",", ":")), prefixNick=False)

    versedump = wrap(versedump, [("checkCapability", "llm.verse.gm"), optional("text")])

    # ------------------------------------------------------------------
    # @versedit <verb> <args...> [#channel]  — operator universe editing
    # ------------------------------------------------------------------

    def versedit(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
        rest: str,
    ) -> None:
        """[#channel] <verb> <args...>

        Edit the verse universe: add, pin, unpin, set, name, desc, retire,
        restore, relate, unrelate, event, editevent, delevent, show.
        Requires the llm.verse.edit capability. A leading #channel targets
        that channel (useful in a PM, to avoid flooding the channel itself);
        otherwise the channel the command was run in is used.
        """
        if self._is_old_message(msg):
            return
        # A leading "#channel" token lets an operator target a channel from a
        # private message — where msg carries no channel of its own (msg.channel
        # is None and msg.args[0] is the bot's nick) — so a batch of edits can be
        # pasted into a DM without flooding the channel. Explicit beats implicit.
        channel_arg: str | None = None
        parts = rest.split(None, 1)
        if parts and ircutils.isChannel(parts[0]):
            channel_arg = parts[0]
            rest = parts[1] if len(parts) > 1 else ""
        channel = channel_arg or (msg.args[0] if msg.args else None)
        if not channel or not ircutils.isChannel(channel):
            irc.error("Specify a channel.", prefixNick=False)
            return
        store = self._get_or_create_verse_store(channel)
        tokens = rest.split(None, 1)
        verb = tokens[0].lower() if tokens else ""
        body = tokens[1] if len(tokens) > 1 else ""
        try:
            reply = self._versedit_dispatch(store, verb, body)
        except (LookupError, ValueError, PermissionError) as exc:
            irc.error(str(exc), prefixNick=False)
            return
        irc.reply(reply, prefixNick=False)

    versedit = wrap(
        versedit,
        [
            # NB: ("checkCapability", ...) checks the cap GLOBALLY against
            # msg.prefix — it does NOT channel-scope to the optional("channel")
            # arg. llm.verse.edit is
            # a global cap everywhere in this plugin; a per-channel grant would
            # fail closed (deny), never escalate. Do not assume per-channel
            # enforcement exists here.
            ("checkCapability", "llm.verse.edit"),
            # "text" is greedy, so a trailing optional("channel") converter
            # could never fire — the leading-#channel token is parsed out of
            # the text in the body instead.
            "text",
        ],
    )

    def _versedit_dispatch(self, store: VerseStore, verb: str, body: str) -> str:
        if verb == "add":
            kind_rest = body.split(None, 1)
            if len(kind_rest) < 2:
                raise ValueError("usage: versedit add <kind> <name> [:: summary]")
            kind, name_part = kind_rest[0], kind_rest[1]
            name, summary = (name_part.split("::", 1) + [""])[:2]
            name, summary = name.strip(), summary.strip()
            if kind not in ("avatar", "npc", "place", "faction", "item"):
                raise ValueError("kind must be avatar|npc|place|faction|item")
            if store.active_name_exists(name):
                raise ValueError(f"an active entity named {name!r} already exists")
            new_id = store.apply_direct(
                op="add_entity",
                payload={"kind": kind, "name": name, "summary": summary},
                source="operator",
                provenance="@versedit add",
            )
            return f"added {kind} #{new_id}: {name}"
        if verb in ("pin", "unpin"):
            eid = store.resolve_ref(body.strip())
            store.apply_direct(
                op="set_pinned",
                payload={"entity_id": eid, "pinned": verb == "pin"},
                source="operator",
                provenance=f"@versedit {verb}",
            )
            return f"{verb}ned #{eid}"
        if verb == "set":
            parts = body.split(None, 2)
            if len(parts) < 3:
                raise ValueError("usage: versedit set <ref> <key> <value>")
            eid = store.resolve_ref(parts[0])
            store.apply_direct(
                op="set_attribute",
                payload={"entity_id": eid, "key": parts[1], "value": parts[2]},
                source="operator",
                provenance="@versedit set",
            )
            return f"set {parts[1]} on #{eid}"
        if verb == "name":
            ref, _, newname = body.partition(" ")
            newname = newname.strip()
            if not newname:
                raise ValueError("usage: versedit name <ref> <new-name>")
            eid = store.resolve_ref(ref)
            if store.active_name_exists(newname):
                raise ValueError(f"an active entity named {newname!r} already exists")
            store.apply_direct(
                op="update_entity",
                payload={"entity_id": eid, "name": newname},
                source="operator",
                provenance="@versedit name",
            )
            return f"renamed #{eid} -> {newname}"
        if verb == "desc":
            ref, _, summary = body.partition("::")
            eid = store.resolve_ref(ref.strip())
            store.apply_direct(
                op="update_entity",
                payload={"entity_id": eid, "summary": summary.strip()},
                source="operator",
                provenance="@versedit desc",
            )
            return f"updated summary of #{eid}"
        if verb in ("retire", "restore"):
            eid = store.resolve_ref(body.strip())
            status = "retired" if verb == "retire" else "active"
            store.apply_direct(
                op="set_status",
                payload={"entity_id": eid, "status": status},
                source="operator",
                provenance=f"@versedit {verb}",
            )
            return f"{verb}d #{eid}"
        if verb == "relate":
            head, _, note = body.partition("::")
            parts = head.split()
            if len(parts) != 3:
                raise ValueError("usage: versedit relate <ref> <kind> <ref> [:: note]")
            from_id = store.resolve_ref(parts[0])
            to_id = store.resolve_ref(parts[2])
            rid = store.apply_direct(
                op="add_relation",
                payload={
                    "from_id": from_id,
                    "to_id": to_id,
                    "kind": parts[1],
                    "note": note.strip(),
                },
                source="operator",
                provenance="@versedit relate",
            )
            return f"related #{from_id} -{parts[1]}-> #{to_id} (relation #{rid})"
        if verb == "unrelate":
            rid = int(body.strip())
            store.apply_direct(
                op="delete_relation",
                payload={"relation_id": rid},
                source="operator",
                provenance="@versedit unrelate",
            )
            return f"deleted relation #{rid}"
        if verb == "event":
            summary, _, ids_part = body.partition("@")
            entity_ids = (
                [int(x) for x in ids_part.split(",") if x.strip().isdigit()] if ids_part else []
            )
            new_id = store.apply_direct(
                op="add_event",
                payload={"summary": summary.strip(), "entity_ids": entity_ids},
                source="operator",
                provenance="@versedit event",
            )
            return f"added event #{new_id}"
        if verb == "editevent":
            id_part, _, summary = body.partition("::")
            ev_id = int(id_part.strip())
            store.apply_direct(
                op="edit_event",
                payload={"event_id": ev_id, "summary": summary.strip()},
                source="operator",
                provenance="@versedit editevent",
            )
            return f"edited event #{ev_id}"
        if verb == "delevent":
            ev_id = int(body.strip())
            store.apply_direct(
                op="delete_event",
                payload={"event_id": ev_id},
                source="operator",
                provenance="@versedit delevent",
            )
            return f"deleted event #{ev_id}"
        if verb == "show":
            eid = store.resolve_ref(body.strip())
            ent = store.get_entity(eid)
            if ent is None:
                raise LookupError(f"entity #{eid} does not exist")
            attrs = store.list_attributes(eid)
            return f"#{eid} {ent.kind} {ent.name} [{ent.status}] — {ent.summary} | attrs={attrs}"
        raise ValueError(f"unknown verb: {verb!r}")

    # ------------------------------------------------------------------
    # @versepurge #chan [token]  — wipe a channel's verse with 2-step confirm
    # ------------------------------------------------------------------

    def _versepurge_check_token(
        self,
        channel: str,
        presented: str,
        now_func: Callable[[], float] = time.time,
    ) -> bool:
        """Return True and clear token if presented token is valid and unexpired.

        Clears expired tokens on False return.  Uses secrets.compare_digest
        to avoid timing side-channels.
        """
        # Lock the check-and-clear so two confirmations can't both consume
        # the same token (the read, compare, and delete must be atomic).
        with self._versepurge_tokens_lock:
            entry = self._versepurge_tokens.get(channel)
            if entry is None:
                return False
            stored_token, expires_at = entry
            if now_func() >= expires_at:
                # Expired — clear stale entry.
                del self._versepurge_tokens[channel]
                return False
            if secrets.compare_digest(stored_token, presented):
                del self._versepurge_tokens[channel]
                return True
            return False

    def versepurge(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
        text: str | None = None,
    ) -> None:
        """[#channel] [<token>]

        Wipe the verse store for a channel. Two-step confirmation required.

          Step 1: @versepurge #chan
                  — issues a one-time 6-character token valid for 60 seconds.

          Step 2: @versepurge #chan <token>
                  — confirms; purges all verse data for the channel.

        Tokens are per-channel and in-memory only; they reset on plugin
        reload or bot restart. All times are approximate (IRC scheduler thread).
        Requires the llm.verse.gm capability.
        """
        # Parse channel and optional token from free-text args.
        raw = (text or "").split()
        channel: str | None = None
        token_presented: str | None = None

        # Order-independent: "@versepurge abc123 #chan" and
        # "@versepurge #chan abc123" both work.
        for token in raw:
            if ircutils.isChannel(token) and channel is None:
                channel = token
            elif token_presented is None:
                token_presented = token

        if channel is None:
            ch = msg.args[0] if msg.args else None
            if ch and ircutils.isChannel(ch):
                channel = ch
            else:
                irc.error("Specify a channel: @versepurge #chan", prefixNick=False)
                return

        if token_presented is not None:
            # Step 2: confirm purge.
            if self._versepurge_check_token(channel, token_presented):
                # Pop the store AND unlink its files under the SAME lock. The
                # plugin is threaded=True, so verse @ask (a SupyThread) and
                # executor workers hold their own thread-local conns — the
                # scheduler thread is NOT the sole holder. Both this block and
                # _get_or_create_verse_store take _verse_stores_lock, so holding
                # it across the unlink prevents a concurrent cache-miss from
                # reconstructing a store on the files mid-purge and leaving a
                # half-written DB behind. (A worker that captured the store
                # before the purge still holds its own conn; its in-flight write
                # is best-effort and may be lost — acceptable for this admin-only,
                # token-gated destructive command.)
                # Stores open lazily, so after a restart the cache misses and
                # the path must be computed from the channel.
                with self._verse_stores_lock:
                    store = self._verse_stores.pop(ircutils.toLower(channel), None)
                    db_path = (
                        Path(store.path)
                        if store is not None
                        else db_path_for_channel(self._verse_data_dir(), channel)
                    )
                    db_path.unlink(missing_ok=True)
                    db_path.with_suffix(".db-wal").unlink(missing_ok=True)
                    db_path.with_suffix(".db-shm").unlink(missing_ok=True)
                irc.reply(f"Verse for {channel} purged.", prefixNick=False)
            else:
                irc.reply(
                    f"Token expired or invalid. Run @versepurge {channel} again to start over.",
                    prefixNick=False,
                )
        else:
            # Step 1: issue (or reissue) token. Lock the read-modify-write so
            # two concurrent issuances can't clobber each other's token.
            new_token = secrets.token_hex(3)
            with self._versepurge_tokens_lock:
                existing = self._versepurge_tokens.get(channel)
                reissued = False
                if existing is not None:
                    _, exp = existing
                    if time.time() < exp:
                        # Reissue while old token still valid — invalidate old one.
                        reissued = True
                    # Old token expired — fall through to fresh issue.
                self._versepurge_tokens[channel] = (new_token, time.time() + 60.0)
            if reissued:
                irc.reply(
                    f"Previous token invalidated. Confirm with @versepurge {channel}"
                    f" {new_token} within 60s.",
                    prefixNick=False,
                )
            else:
                irc.reply(
                    f"Confirm with @versepurge {channel} {new_token} within 60s.",
                    prefixNick=False,
                )

    versepurge = wrap(versepurge, [("checkCapability", "llm.verse.gm"), optional("text")])

    # ------------------------------------------------------------------
    # @versecompact — manual retention compaction (E4)
    # ------------------------------------------------------------------

    def versecompact(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
        channel: str,
    ) -> None:
        """<channel>

        Manually run retention compaction for <channel>. Mirrors what the
        daily timer does (``_run_compaction_pass``) but for one channel only.
        Requires capability llm.verse.gm.
        """
        if not self.registryValue("verseEnabled", channel):
            irc.reply(
                f"verseEnabled is False for {channel}; nothing to compact.",
                prefixNick=False,
            )
            return

        min_keep, model, client = self._compaction_settings()
        try:
            # Inside the executor cap like every other blocking LLM call —
            # this used to be the only one that bypassed it.
            with self._allow_concurrent(), self._llm_executor.permit():
                outcome, _ = self._compact_channel(
                    channel, min_keep=min_keep, model=model, client=client, run_aging=False
                )
        except Exception as exc:
            self.log.exception("@versecompact failed for %s", channel)
            irc.error(
                f"compaction failed for {channel}: {type(exc).__name__}",
                prefixNick=False,
            )
            return

        outcome_msg = _format_compaction_outcome(outcome, None, min_keep_events=min_keep)
        irc.reply(
            f"compaction outcome for {channel}: {outcome_msg}",
            prefixNick=False,
        )

    versecompact = wrap(
        versecompact,
        [
            ("checkCapability", "llm.verse.gm"),
            "channel",
        ],
    )


Class = LLM
