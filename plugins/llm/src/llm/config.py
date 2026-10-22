"""Configuration for LLM plugin."""

from __future__ import annotations

import logging
import threading

import supybot.conf as conf
import supybot.registry as registry
from supybot.i18n import PluginInternationalization

from . import apikeys, provider

_ = PluginInternationalization("LLM")

_log = logging.getLogger("supybot.plugins.LLM.config")


class ValidatedModelName(registry.String):
    """A ``provider/name`` model string for a provider the plugin can call.

    - Empty strings are accepted (means "not configured").
    - Any other prefix is rejected: provider.py has no client for it.
    - A model with no price in provider.py is accepted with a one-time
      warning, because its spend would book as $0.
    """

    _warned: set[str] = set()
    _warned_lock: threading.Lock = threading.Lock()

    def setValue(self, v: str) -> None:  # noqa: N802
        v = v.strip()
        if v:
            self._validate_model(v)
        super().setValue(v)

    def _validate_model(self, model: str) -> None:
        if not apikeys.provider_of(model):
            prefixes = ", ".join(f"{p}/" for p in sorted(apikeys.PROVIDER_ENV_VARS))
            raise registry.InvalidRegistryValue(
                f"Unknown model: {model!r}. Use a {prefixes} model."
            )
        if model in provider.PRICES or model in provider.IMAGE_PRICES:
            return
        with ValidatedModelName._warned_lock:
            if model not in ValidatedModelName._warned:
                ValidatedModelName._warned.add(model)
                _log.warning(
                    "Model %r has no price in provider.py; its spend will book as $0.", model
                )


def configure(advanced: bool) -> None:
    """Plugin configuration wizard."""
    conf.registerPlugin("LLM", True)

    print("=" * 60)
    print("LLM Plugin Configuration")
    print("=" * 60)
    print("\nThis plugin provides AI-powered commands for Anthropic and xAI models.")
    print("You'll need API keys for the models you want to use.")
    print("\nAPI keys come from the environment, one per provider:")
    print("  XAI_API_KEY, ANTHROPIC_API_KEY")
    print("\nFor more info, see the README.md")
    print("=" * 60)


LLM = conf.registerPlugin("LLM")

# ============================================================================
# System Prompts (channel-specific with global defaults)
# ============================================================================

conf.registerChannelValue(
    LLM,
    "assistantSystemPrompt",
    registry.String(
        "You are a helpful IRC assistant. This is IRC chat — replies are read "
        "in a terminal client, not rendered as markdown. Keep answers tight: "
        "lead with the answer, aim for one line, never exceed three. Plain "
        "text only — no bold, italics, headings, backticks, code fences, "
        "bullet lists, or [label](url) links. Write URLs bare.",
        _("""System prompt for all assistant work - defines bot personality and constraints."""),
    ),
)

# ============================================================================
# Model Configuration (channel-specific with global defaults)
# ============================================================================

conf.registerChannelValue(
    LLM,
    "assistantModel",
    ValidatedModelName(
        "anthropic/claude-haiku-5-5",
        _("""Model used for all assistant text+tool work (chat, planner loop,
        memory, reminder parsing, scheduled tasks). Must support
        vision if image URLs in chat should work."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "memoryModel",
    ValidatedModelName(
        "",
        _("""Model for the memory extraction that runs after every reply.
        Falls back to assistantModel if empty. Its output is never shown, so a
        small model is enough."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "verseModel",
    ValidatedModelName(
        "",
        _("""Model for verse-mode (in-world roleplay) replies. Falls back to
        assistantModel if empty. Useful when the channel's assistantModel is a
        reasoning model that empirically produces terse output for long-form
        scenes — point this at a non-reasoning model to get richer prose for
        verse turns without changing chat-mode behavior."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "verseStyleExemplars",
    registry.Json(
        [],
        _("""Curated "taste" style exemplars — a JSON list of strings injected
        into the verse system prompt to bias prose toward what the channel's
        critics like. Default empty = verse prompt unchanged. Populated offline
        by mining channel logs (plugins/llm/src/llm/verse/taste_mine.py) and
        curated by hand; deploy by editing this value in bot.conf while the bot
        is stopped (a JSON array survives the round-trip cleanly)."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "drawAutoRewriteMax",
    registry.NonNegativeInteger(
        # 1, not the 3 this defaulted to while the loop was unreachable on the
        # prod image model. Every rewrite spends a BILLED image call on top of
        # the one already refused, and the recoveries observed on 2026-08-15 all
        # landed on the first changed prompt. A cap of 1 buys the bulk of them
        # for one extra call; 3 mostly buys three refusals of a prompt the
        # provider was never going to pass.
        1,
        _("""Maximum number of automatic prompt rewrites when image generation
        is blocked by content safety filters. Set to 0 to disable. Each retry
        uses the ask model to rewrite the prompt, then spends another billed
        image call."""),
    ),
)

# ============================================================================
# Verse (forest-verse avatar/event subsystem)
# ============================================================================

conf.registerChannelValue(
    LLM,
    "verseEnabled",
    registry.Boolean(
        False,
        _("""Enable the verse avatar/event subsystem in this channel.
        When False, @verse commands are disabled and no verse events are
        recorded regardless of other verse settings."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "verseTriggerRegex",
    registry.String(
        r"\bverse\b",
        _("""Regex (case-insensitive, re.search) that force-routes a message
        into verse mode when it matches. This is in ADDITION to the always-on
        signal that a message naming a known in-universe entity (character,
        place, or item, other than the speaker's own avatar) triggers verse.
        Empty string disables the keyword trigger, leaving only entity
        references. A malformed regex is ignored (treated as no match).
        Non-triggering messages fall through to the normal chat path."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "verseRosterMaxChars",
    registry.PositiveInteger(
        4000,
        _("""Max characters of the canon-roster block (pinned or author-locked
        entities) injected into every verse system prompt. Canon entities
        beyond the cap are dropped with a (roster truncated) marker."""),
    ),
)

# ============================================================================
# Storybook (illustrated story tool) — despite the verse* prefix these keys
# also govern the standalone @story command, which needs no verse. The
# prefix is kept because renaming registry keys silently orphans prod
# bot.conf values.
# ============================================================================

conf.registerChannelValue(
    LLM,
    "verseStorybookEnabled",
    registry.Boolean(
        False,
        _("""Expose the verse_storybook tool in this channel. (@story is
        independent of this flag — it is gated like @draw.)"""),
    ),
)

conf.registerChannelValue(
    LLM,
    "verseChatRecordEnabled",
    registry.Boolean(
        False,
        _("""Let canon accrue from ORDINARY chat, not just @rp roleplay turns.
        When on, an opted-in avatar's normal (non-roleplay) chat turn can call
        verse_record to save a genuinely new, durable canon fact; the model is
        nudged to do so sparingly. Off (default) keeps chat-path verse_record
        denied, so canon only grows during explicit roleplay. Guards
        canon-pollution risk behind an opt-in per the cautious verse-write
        rollout pattern."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "verseStorybookMaxImages",
    registry.NonNegativeInteger(
        5,
        _("""Max illustrations drawn per story (verse_storybook and @story)."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "verseStorybookCooldownSeconds",
    registry.NonNegativeInteger(
        300,
        _("""Per-account cooldown between stories (shared by verse_storybook
        and @story)."""),
    ),
)

# ============================================================================
# HTTP Server Settings (for image/answer output)
# ============================================================================

conf.registerGlobalValue(
    LLM,
    "httpRoot",
    registry.String(
        "",
        _("""Filesystem path to save image and page files. If empty, uses
        Limnoria's web directory (data/web/llm/) and built-in HTTP server.
        If set, files are saved there and Limnoria's HTTP server is NOT
        used (external server like nginx expected)."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "httpUrlBase",
    registry.String(
        "",
        _("""Base URL for accessing saved files. If empty, uses Limnoria's
        HTTP server publicUrl + /llm/. Example: https://example.com/llm"""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "imageUploadUrl",
    registry.String(
        "",
        _("""Endpoint of an external image host for generated images. Expects a
        multipart POST on the "images[]" field and a JSON reply shaped like
        {"results": [{"success": true, "filePath": "/img/x.png"}]} — for example
        https://paste.boxlabs.uk/img/. If empty, images are stored locally under
        httpRoot and served from httpUrlBase. Any upload failure falls back to
        local storage, so this is safe to leave pointed at a flaky host."""),
    ),
)

# ============================================================================
# Conversation Context (channel-specific with global defaults)
# ============================================================================

conf.registerChannelValue(
    LLM,
    "contextTimeoutMinutes",
    registry.PositiveInteger(
        5,
        _("""Clear context after this many minutes of inactivity"""),
    ),
)

conf.registerChannelValue(
    LLM,
    "privacyOptInOnly",
    registry.Boolean(
        True,
        _("""Privacy: when True, only lines addressed to the bot (and its own
        replies) reach the LLM provider. Set False to also feed every channel
        message into the shared context and let irc_lookup read the channel's
        server history (kind=history) for 'what did I miss'. Off, those lines
        go to whichever provider serves this channel's models; turn it off only
        when all of them are local or the channel has consented."""),
    ),
)

# ============================================================================
# Advanced Settings
# ============================================================================

conf.registerGlobalValue(
    LLM,
    "timeout",
    registry.PositiveInteger(
        30,
        _("""Timeout for LLM API calls in seconds"""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "fileCleanupAge",
    registry.PositiveInteger(
        720,
        _("""Delete HTTP files older than this many hours (default: 720 = 30 days)"""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "commandPrefixes",
    registry.SpaceSeparatedListOfStrings(
        [".", "@"],
        _("""Command prefixes to sanitize in output. Lines starting with these
        are prefixed with a space to prevent IRC command injection.
        Default: . @"""),
    ),
)

# ============================================================================
# Rate Limiting (per-command, per-tier)
#
# Tiers (checked from most to least privileged):
#   owner/admin  — always exempt (no config needed)
#   trusted      — relaxed limits (Trusted prefix)
#   registered   — standard limits (no prefix)
#   unregistered — strictest limits (Unreg prefix), ask only: draw, meme and
#                  story refuse an unauthenticated caller before the rate
#                  limiter runs, so that tier never reaches them
#
# Setting any count to 0 disables rate limiting for that command+tier.
# ============================================================================

_TIER_LABELS = {
    "": "registered tier",
    "Trusted": "trusted tier",
    "Unreg": "unregistered tier",
}


def _register_rate_limit_block(command: str, *, limits: dict[str, tuple[int, int]]) -> None:
    """Register ``{command}{tier}RateLimit{Count,Window}`` for each tier in ``limits``.

    ``limits`` maps a tier infix ("" for registered, "Trusted", "Unreg") to
    its (count, window) defaults.
    """
    for tier, (count, window) in limits.items():
        label = _TIER_LABELS[tier]
        conf.registerGlobalValue(
            LLM,
            f"{command}{tier}RateLimitCount",
            registry.NonNegativeInteger(
                count,
                _(
                    f"Max {command} requests per {label} within "
                    f"{command}{tier}RateLimitWindow seconds. "
                    "Set to 0 to disable rate limiting for this tier."
                ),
            ),
        )
        conf.registerGlobalValue(
            LLM,
            f"{command}{tier}RateLimitWindow",
            registry.PositiveInteger(
                window,
                _(f"Time window in seconds for counting {command} requests ({label})."),
            ),
        )


# --- ask (cheapest; the only command open to unauthenticated callers) ---
_register_rate_limit_block(
    "ask",
    limits={"": (15, 60), "Trusted": (15, 60), "Unreg": (15, 60)},
)

# --- draw (expensive) ---
_register_rate_limit_block(
    "draw",
    limits={"": (2, 300), "Trusted": (5, 60)},
)

# --- meme (free: rendered locally) ---
# The ceiling is politeness to the channel, not spend.
_register_rate_limit_block(
    "meme",
    limits={"": (6, 60), "Trusted": (12, 60)},
)

# --- story (expensive, illustrated storybook) ---
_register_rate_limit_block(
    "story",
    limits={"": (2, 300), "Trusted": (5, 60)},
)

# ============================================================================
# Bot-to-bot loop guard
# ============================================================================

conf.registerGlobalValue(
    LLM,
    "botLoopHostSuffixes",
    registry.SpaceSeparatedListOfStrings(
        ["Bot.AfterNET.Org"],
        _("""Host suffixes the network reserves for registered bots. Matched
        case-insensitively against the host part of the sender's hostmask, and
        treated as proof the sender is a bot. This carries the guard where the
        +B user mode cannot: grok never sets the mode but does live on
        grok.Bot.AfterNET.Org. A bot matched here, or flagged +B, also cannot
        address the bot by replying to its lines. A full host works as a
        suffix, for a bot on a user host. Empty the list to rely on +B
        alone."""),
    ),
)

# ============================================================================
# Memes (@meme / make_meme tool)
# ============================================================================

conf.registerGlobalValue(
    LLM,
    "memeAliases",
    registry.SpaceSeparatedListOfStrings(
        ["tyrone=yallgot", "biggums=yallgot", "chappelle=yallgot"],
        _("""Extra names for meme templates, as name=target pairs. A target
        that is a template id makes the name a keyword on it (Tyrone Biggums
        is filed under the quote, yallgot). A target
        that is an image URL adds a two-caption template drawn on that image.
        Use hyphens for spaces in the name."""),
    ),
)

# ============================================================================
# Limnoria tool bridge (Phase 1)
# ============================================================================

conf.registerChannelValue(
    LLM,
    "bridgeEnabled",
    registry.Boolean(
        False,
        _("""When True, expose loaded Limnoria plugin commands to the LLM
        as a tool, restricted by bridgeAllowedPlugins and Limnoria's
        capability system. Default off."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "bridgeAllowedPlugins",
    registry.SpaceSeparatedListOfStrings(
        [],
        _("""Space-separated list of Limnoria plugin names whose commands
        the LLM may call when bridgeEnabled is True. When empty (the
        default) the bridge falls back to a curated read-safe set: Misc
        Time Math Utilities Seen Web Later Note Karma QuoteGrabs RSS DDG.
        Write commands are never exposed, whichever plugins are listed
        (MUTATING_COMMANDS in limnoria_bridge.py). Set this to a non-empty list to
        override the curated set with your own selection. Set
        bridgeEnabled False to disable the bridge entirely."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "pendingTasksEnabled",
    registry.Boolean(
        False,
        _("""When True, chat completions in this channel advertise the
        reminder/scheduled-task tools (set_reminder, schedule_llm_task,
        list_pending_tasks, cancel_pending_task, cancel_all_pending_tasks)
        and their operating rules, so natural-language scheduling via @ask
        works ('remind me in 10 minutes', 'every hour post X').

        Off by default: the five tool schemas plus the prompt rules cost
        roughly 1,100 prompt tokens on EVERY completion, whether or not
        anyone schedules anything. The @remind command and the firing of
        already-created reminders/tasks work regardless of this setting —
        it only gates whether the model can create/list/cancel them."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "bridgeDebugInChannel",
    registry.Boolean(
        False,
        _("""When True, append a one-line debug footer to LLM replies listing
        every Limnoria bridge tool call made during the turn (plugin, command,
        args, ok/err). Off by default; useful to confirm the LLM is actually
        using bridged commands."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "statusPageUrls",
    registry.SpaceSeparatedListOfStrings(
        [
            "Claude=https://status.claude.com",
            "GitHub=https://www.githubstatus.com",
            "OpenAI=https://status.openai.com",
        ],
        _("""Space-separated status pages to poll and announce, each written as
        Name=url or as a bare url (which takes its host as its name). Both
        Atlassian Statuspage and incident.io pages work. Name is 1-32 chars of
        [A-Za-z0-9._-] and is what the model uses to ask about one service. The
        url must be a bare scheme://host with no trailing path. Unusable
        entries are dropped with a warning, duplicates collapse, and at most 5
        are polled. Empty disables polling and announcements."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "auroraAnnounce",
    registry.Boolean(
        False,
        _("""Announce NOAA G3+ geomagnetic storm watches, and sustained storm
        conditions in the solar wind, in this channel. Off by default."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "auroraAnnouncePlaces",
    registry.SpaceSeparatedListOfStrings(
        [],
        _("""Optional places for this channel's aurora announcements; _ stands
        for a space (Sault_Ste._Marie). Adds cloud cover for them, and holds a
        storm line until one of them is dark. Empty means storm lines go out at
        once with no cloud figures."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "statusAnnounce",
    registry.Boolean(
        False,
        _("""Announce status-page incidents in this channel as they open and
        again as they resolve. Off by default. When enabling this, remove the
        equivalent RSS feed announcement for the channel first, or every
        incident is reported twice."""),
    ),
)
