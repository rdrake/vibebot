"""Configuration for LLM plugin."""

from __future__ import annotations

import difflib
import logging
import threading

import litellm
import supybot.conf as conf
import supybot.registry as registry
from supybot.i18n import PluginInternationalization

_ = PluginInternationalization("LLM")

_log = logging.getLogger("supybot.plugins.LLM.config")

# refusalFallbackModel's default. LiteLLM's price table has no entry for it, so
# completion_cost would fail and its spend would book at $0. The alias is
# served by grok-4.3 non-reasoning (owner, 2026-10-14), so it is priced at
# grok-4.3's per-token rates. Registered here rather than in service.py because
# the registry validates defaults at import, and config loads first.
REFUSAL_FALLBACK_MODEL = "xai/grok-4-1-fast-non-reasoning"
litellm.register_model(
    {
        REFUSAL_FALLBACK_MODEL: {
            key: litellm.model_cost["xai/grok-4.3"][key]
            for key in (
                "input_cost_per_token",
                "output_cost_per_token",
                "cache_read_input_token_cost",
            )
        }
        | {"litellm_provider": "xai", "mode": "chat"}
    }
)


class ValidatedModelName(registry.String):
    """A model name validated against litellm's known models.

    - Empty strings are accepted (means "not configured").
    - Models that litellm can parse (provider recognized) are accepted.
      If the specific model isn't in litellm.model_list, a warning is logged once.
    - Models that litellm cannot parse at all are rejected with suggestions.
    """

    _warned: set[str] = set()
    _warned_lock: threading.Lock = threading.Lock()

    def setValue(self, v: str) -> None:  # noqa: N802
        v = v.strip()
        if v:
            self._validate_model(v)
        super().setValue(v)

    def _validate_model(self, model: str) -> None:
        """Validate a model name against litellm's known models."""
        try:
            litellm.get_llm_provider(model)
        except Exception as exc:  # noqa: BLE001
            # BadRequestError is litellm's documented "unknown model" signal,
            # but get_llm_provider can also raise other types internally (e.g.
            # a KeyError/ValueError on a malformed provider prefix). Any of
            # those escaping here would abort registry construction / bot.conf
            # loading at startup instead of rejecting one bad value, so map
            # every failure to a clean InvalidRegistryValue.
            suggestions = self._suggest_models(model)
            msg = f"Unknown model: {model!r}."
            if suggestions:
                msg += f" Did you mean: {', '.join(suggestions)}?"
            else:
                msg += " See https://docs.litellm.ai/docs/providers for supported models."
            if not isinstance(exc, litellm.exceptions.BadRequestError):
                msg += f" ({type(exc).__name__})"
            raise registry.InvalidRegistryValue(msg) from exc

        # Provider recognized — warn once if model is not in the known list.
        # model_cost also counts: litellm.register_model fills only that.
        if model not in litellm.model_list and model not in litellm.model_cost:
            with ValidatedModelName._warned_lock:
                if model not in ValidatedModelName._warned:
                    ValidatedModelName._warned.add(model)
                    suggestions = self._suggest_models(model)
                    hint = ""
                    if suggestions:
                        hint = f" Similar known models: {', '.join(suggestions)}."
                    _log.warning(
                        "Model %r not in litellm's known model list (may be a custom "
                        "deployment or newer model).%s",
                        model,
                        hint,
                    )

    @staticmethod
    def _suggest_models(model: str, n: int = 3, cutoff: float = 0.6) -> list[str]:
        """Suggest similar model names using fuzzy matching."""
        # Try matching against full model list first
        matches = difflib.get_close_matches(model, litellm.model_list, n=n, cutoff=cutoff)
        if matches:
            return matches

        # If provider-prefixed, also try matching within that provider's models
        if "/" in model:
            provider, model_name = model.split("/", 1)
            provider_models = litellm.models_by_provider.get(provider, set())
            if provider_models:
                sub_matches = difflib.get_close_matches(
                    model_name, list(provider_models), n=n, cutoff=cutoff
                )
                return [f"{provider}/{m}" for m in sub_matches]

        return []


_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


class ValidatedLogLevel(registry.String):
    """A log level name validated against Python's standard levels."""

    def setValue(self, v: str) -> None:  # noqa: N802
        v = v.strip().upper()
        if v not in _VALID_LOG_LEVELS:
            raise registry.InvalidRegistryValue(
                f"Invalid log level: {v!r}. Must be one of: {', '.join(sorted(_VALID_LOG_LEVELS))}"
            )
        super().setValue(v)


def configure(advanced: bool) -> None:
    """Plugin configuration wizard."""
    conf.registerPlugin("LLM", True)

    print("=" * 60)
    print("LLM Plugin Configuration")
    print("=" * 60)
    print("\nThis plugin provides AI-powered commands using LiteLLM.")
    print("You'll need API keys for the models you want to use.")
    print("\nAPI keys come from the environment, one per provider:")
    print("  XAI_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY")
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

conf.registerChannelValue(
    LLM,
    "codeSystemPrompt",
    registry.String(
        "You are a helpful code assistant. Explain your code and provide context. "
        "Replies go to IRC: keep prose to one or two short lines, plain text, no "
        "markdown. Code itself is delivered via the generate_code tool's URL — "
        "do not paste the code body into the chat reply.",
        _("""System prompt for code command"""),
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
    "refusalFallbackModel",
    ValidatedModelName(
        REFUSAL_FALLBACK_MODEL,
        _("""Model that re-runs an assistant turn once when the turn's model
        refuses: a policy refusal that survived its one retry, or a provider
        refusal error. The fallback sees the original request only, never the
        refusal. Empty disables. Skipped when the turn already runs on this
        model or its provider has no key."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "codeModel",
    ValidatedModelName(
        "anthropic/claude-haiku-5-5",
        _("""Model for code generation"""),
    ),
)

conf.registerChannelValue(
    LLM,
    "imageModel",
    ValidatedModelName(
        "xai/grok-imagine-image",
        _("""Model for image generation"""),
    ),
)

conf.registerChannelValue(
    LLM,
    "drawEnabled",
    registry.Boolean(
        True,
        _("""When False, @draw is refused and chat does not advertise the
        generate_image tool in this channel. The tool is also hidden when
        imageModel's provider has no key configured."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "imageApiBase",
    registry.String(
        "",
        _("""Base URL of an OpenAI-shaped image endpoint to draw against instead
        of the model's own provider, e.g. http://host:port/v1 for the
        self-hosted box. Empty means draw through the provider LiteLLM infers
        from imageModel, which is the hosted default. When set, the bearer
        token is ANIMATE_API_KEY — the self-hosted server's credential — not
        the provider key imageModel's name would imply."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "imageSteps",
    registry.NonNegativeInteger(
        0,
        _("""Denoising steps for a self-hosted image endpoint. 0 leaves the
        choice to the server. Ignored unless imageApiBase is set, since hosted
        providers do not take this parameter. Steps are the dominant cost:
        measured on the reference box at 1024x576, 8 steps took ~27s and 25
        took ~94s on an idle server."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "imageSize",
    registry.String(
        "",
        _("""Output geometry as WxH for a self-hosted image endpoint, e.g.
        1024x576. Empty lets the server choose. Ignored unless imageApiBase is
        set: hosted providers are tuned through their own knobs instead."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "imageFallbackApiBase",
    registry.String(
        "",
        _("""Base URL of an OpenAI-shaped image endpoint to draw against when the
        primary provider REFUSES a prompt on content grounds. Empty disables
        the fallback and a refusal reaches the user as an error, which is the
        behaviour without this set. Authenticates with ANIMATE_API_KEY. The
        fallback is sent the user's ORIGINAL prompt, never a safety rewrite:
        an endpoint that does not refuse has nothing to be talked around."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "imageFallbackModel",
    registry.String(
        "",
        _("""Model for imageFallbackApiBase. A plain string, not a validated
        model name, because a self-hosted server names its models whatever it
        likes — a filesystem path, for instance. Prefix with the wire format
        LiteLLM should speak, e.g. openai//work/models/Some-Model."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "imageFallbackSteps",
    registry.NonNegativeInteger(
        0,
        _("""Denoising steps for the fallback endpoint. 0 lets the server
        choose. See imageSteps: this is the dominant cost of a fallback
        draw."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "imageFallbackSize",
    registry.String(
        "",
        _("""Output geometry as WxH for the fallback endpoint. Empty lets the
        server choose."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "searchModel",
    ValidatedModelName(
        "xai/grok-4.3",
        _("""Model for search/fetch tools. Falls back to assistantModel if empty.
        Only xAI and Gemini models search; any other provider answers from
        its own knowledge without looking anything up."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "memoryModel",
    ValidatedModelName(
        "",
        _("""Model for the memory extraction that runs after every reply.
        Falls back to assistantModel if empty. Its output is never shown, so a
        flash-lite class model is enough; on Gemini a different model also
        keeps these calls out of the chat model's free-tier daily quota."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "subjectResearchEnabled",
    registry.Boolean(
        True,
        _("""Research the real subjects a @draw or @animate request names
        before the planner writes the prompt. One grounded completion looks up
        how the named people, places and events actually look and the facts are
        layered onto the planner's prompt, because the generators know nothing
        by name. Costs one extra call and a few seconds per request: @animate
        hides that behind the render, @draw does not."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "subjectResearchModel",
    ValidatedModelName(
        "",
        _("""Model for the subject-research pre-stage. Falls back to
        searchModel, then assistantModel. Worth setting when the search model
        is more reticent about describing real people than the generator
        downstream is."""),
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
        scenes — point this at a non-reasoning model (e.g. gemini-flash-latest)
        to get richer prose for verse turns without changing chat-mode
        behavior."""),
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

conf.registerGlobalValue(
    LLM,
    "drawTimeout",
    registry.NonNegativeInteger(
        120,
        _("""Timeout for image generation API calls in seconds. Image generation
        is slower than text completion, especially with auto-rewrite retries.
        Set to 0 to use the global timeout setting instead."""),
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

conf.registerChannelValue(
    LLM,
    "drawContextMaxAgeSeconds",
    registry.NonNegativeInteger(
        60,
        _("""Only pass conversation context to draw requests when the
        conversation's last activity is within this many seconds. Keeps
        draw calls grounded in very recent discussion without overwhelming
        the image model with stale history. Set to 0 to disable (always
        start fresh)."""),
    ),
)

# ============================================================================
# Video Generation (animate)
# ============================================================================
# Not a LiteLLM provider: a self-hosted vLLM box exposing an OpenAI-shaped
# /v1/videos job API. Submit returns a job id immediately and the job survives
# client disconnect, so animate never blocks a turn — it stashes the id in
# pending_tasks and the existing 30s poller delivers the link when it lands.
# The bearer token lives in ANIMATE_API_KEY, not here: the registry is
# bot.conf, which the bot rewrites on shutdown, and _API_KEY suffixes are
# already covered by log redaction (apikeys.SECRET_SUFFIXES).

conf.registerChannelValue(
    LLM,
    "animateEnabled",
    registry.Boolean(
        True,
        _("""When False, @animate is refused and chat does not advertise the
        generate_video tool in this channel. Both are also off everywhere
        until animateApiUrl and ANIMATE_API_KEY are set."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "animateApiUrl",
    registry.String(
        "",
        _("""Base URL of the vLLM video server, e.g. http://host:14205 — no
        trailing /v1. Empty disables the animate command and hides the
        generate_video tool. The bearer token is read from the ANIMATE_API_KEY
        environment variable, never from this registry."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "animateModel",
    registry.String(
        "",
        _("""Model to request from the video server. Empty (default) sends no
        model field, letting the server pick the one it has loaded — which is
        the right default for a single-model box."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "animateSize",
    registry.String(
        "1280x704",
        _("""Output resolution as WxH. Must be a geometry the loaded model
        supports; the server rejects anything else."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "animateDuration",
    registry.PositiveInteger(
        7,
        _("""Clip length in seconds. Generation time scales with this, and the
        whole clip is exclusive GPU time, so raising it slows every queued
        request behind it. Watch the file size too: clips over the uploader's
        10 MB ceiling silently fall back to local storage instead of the
        configured image host."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "animateSteps",
    registry.PositiveInteger(
        # 25, not the 50 the endpoint's own example uses. Steps are the main
        # lever on latency and it is close to linear: 50 steps measured ~138s
        # for a 4s clip, which is a long time to make a channel wait. 25 buys
        # most of the quality back at roughly half the wall clock.
        25,
        _("""Denoising steps. The dominant cost knob — latency is roughly linear
        in this. Raise for quality, lower for turnaround."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "animateAudio",
    registry.Boolean(
        True,
        _("""Generate a soundtrack alongside the video (the t2va task). The
        loaded FL2VA model is built for it. False requests silent video."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "animateFlowShift",
    registry.NonNegativeInteger(
        12,
        _("""Flow-matching shift for the video sampler. Tuning knob — leave at
        the default unless the model card says otherwise."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "animateAudioFlowShift",
    registry.NonNegativeInteger(
        3,
        _("""Flow-matching shift for the audio track. Ignored when animateAudio
        is False."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "animateTimeout",
    registry.PositiveInteger(
        60,
        _("""Timeout in seconds for individual HTTP calls to the video server
        (submit, poll, download). NOT the generation budget — generation is
        asynchronous and bounded by animateExpiry instead."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "animateExpiry",
    registry.PositiveInteger(
        1800,
        _("""How long to keep polling a submitted video job before giving up and
        telling the user it expired. Generous by design: a queued job behind
        several others can wait a long time before it starts."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "animateMaxPendingPerUser",
    registry.NonNegativeInteger(
        2,
        _("""How many clips one person may have waiting to render at once.
        A request past this is refused before anything is submitted. Owner
        and admin are exempt. 0 disables the per-user cap."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "animateMaxPending",
    registry.NonNegativeInteger(
        6,
        _("""How many clips may be waiting to render across the whole bot.
        The video box renders one at a time, so this bounds the longest
        wait anyone is promised. Applies to everyone. 0 disables."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "animateRenderSeconds",
    registry.PositiveInteger(
        135,
        _("""Seconds one clip takes to render at the configured size, duration
        and steps. Only used to estimate the wait quoted when a clip is
        queued; measure it on the box after changing animateDuration or
        animateSteps."""),
    ),
)

# ============================================================================
# Memory Extraction
# ============================================================================

conf.registerChannelValue(
    LLM,
    "memoryEnabled",
    registry.Boolean(True, _("""Enable automatic memory extraction from command interactions.""")),
)

conf.registerGlobalValue(
    LLM,
    "memoryMaxPerUser",
    registry.PositiveInteger(50, _("""Maximum number of memories stored per user.""")),
)

conf.registerGlobalValue(
    LLM,
    "memoryCleanupInterval",
    registry.NonNegativeInteger(
        3,
        _("""Number of memory extraction passes (that save at least one fact)
        between automatic cleanup runs. Set to 0 to disable periodic cleanup."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "memoryPromotionThreshold",
    registry.PositiveInteger(
        2,
        _("""Number of times a candidate fact must be reinforced before it
        becomes a durable memory. 1 disables the candidate stage (every
        extraction is saved immediately, restoring legacy behavior)."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "memoryCandidateTTLDays",
    registry.NonNegativeInteger(
        14,
        _("""Days a candidate fact may sit unreinforced before it is pruned.
        Set to 0 to disable TTL pruning."""),
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
    "verseEventRetentionDays",
    registry.NonNegativeInteger(
        30,
        _("""Number of days to retain verse events before pruning.
        Older events are removed during housekeeping runs. Must be
        non-negative: a negative value would push the prune cutoff into
        the future and delete every event."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "verseAutoEntityRetireDays",
    registry.NonNegativeInteger(
        14,
        _("""Days of no reference before auto-created NPCs retire. 0 disables sweep."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "verseAutoEntityMaxNamesPerCall",
    registry.PositiveInteger(
        8,
        _("""Hard cap on verse_record `actors` array length. The advertised
        tool spec's maxItems is set from this; dispatch enforces. Increase
        past 16 only if your verse routinely cites large casts."""),
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

conf.registerGlobalValue(
    LLM,
    "verseCompactionModel",
    registry.String(
        "anthropic/claude-haiku-5-5",
        _("""Model used by the verse compaction job to summarise old events
        into a lore digest. Mirrors the default of the old
        ``loomModel`` key; split out when the loom was removed so compaction
        can be configured independently."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "verseCompactionDailyAt",
    registry.String(
        "03:00",
        _("""Local-time HH:MM at which the daily verse-event-retention
        compaction job fires. Empty or malformed values defer the next
        run by one hour."""),
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
    "verseRoleplayStickyTtlSeconds",
    registry.NonNegativeInteger(
        900,
        _("""Sliding inactivity timeout (seconds) for sticky @rp roleplay mode
        (`@rp on`). Each in-character turn refreshes it; after this much silence
        you drop back to normal chat. 0 = never auto-expire (until `@rp off` or
        a bot restart)."""),
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

# (Verse engagement measurement, not storybook — filed here historically.)
conf.registerChannelValue(
    LLM,
    "verseReactionCaptureEnabled",
    registry.Boolean(
        True,
        _("""Capture inbound IRCv3 emoji reactions (+draft/react) to the bot's
        verse lines as an offline approval signal (reactions.jsonl). Recency-
        attributed; measurement only — no behaviour change. Kill-switch."""),
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
    "verseStorybookMaxPerTurn",
    registry.NonNegativeInteger(
        1,
        _("""Max verse_storybook calls honored per completion."""),
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

conf.registerChannelValue(
    LLM,
    "verseStorybookDailyImageCap",
    registry.NonNegativeInteger(
        30,
        _("""Per-account daily ceiling on storybook images (enforcement
        deferred; cooldown + per-turn cap bound spend today)."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "verseStorybookMaxChars",
    registry.NonNegativeInteger(
        6000,
        _("""Story length cap before image embedding (verse_storybook and
        @story)."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "verseStorybookImageTimeout",
    registry.NonNegativeInteger(
        45,
        _("""Per-image timeout (seconds) for storybook draws (verse_storybook
        and @story)."""),
    ),
)

# (Verse retention compaction, not storybook — filed here historically.)
conf.registerGlobalValue(
    LLM,
    "verseCompactionMinKeepEvents",
    registry.NonNegativeInteger(
        20,
        _("""Floor on total event count below which a verse is left
        alone by compaction. Prevents thrashing small verses."""),
    ),
)

# ============================================================================
# Pending Task Retry (per-command expiry)
# ============================================================================

conf.registerGlobalValue(
    LLM,
    "askExpiry",
    registry.NonNegativeInteger(
        60,
        _("""Maximum seconds to keep retrying timed-out ask requests.
        Set to 0 to disable background retry for ask."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "codeExpiry",
    registry.NonNegativeInteger(
        60,
        _("""Maximum seconds to keep retrying timed-out code requests.
        Set to 0 to disable background retry for code."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "drawExpiry",
    registry.NonNegativeInteger(
        60,
        _("""Maximum seconds to keep retrying timed-out draw requests.
        Set to 0 to disable background retry for draw."""),
    ),
)

# ============================================================================
# HTTP Server Settings (for code/image/answer output)
# ============================================================================

conf.registerGlobalValue(
    LLM,
    "httpRoot",
    registry.String(
        "",
        _("""Filesystem path to save code/image files. If empty, uses
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

conf.registerGlobalValue(
    LLM,
    "helpUrl",
    registry.String(
        "https://rdrake.github.io/vibebot-v8/",
        _("""URL to the help documentation page. Shown in plugin help output."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "longReplyTeaserMaxChars",
    registry.PositiveInteger(
        220,
        _("""Maximum characters for the one-line teaser shown in IRC when a
        long reply is linked to the full HTML answer."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "multilineMaxLines",
    registry.NonNegativeInteger(
        3,
        _("""Replies that wrap to at most this many IRC lines are sent inline as
        one IRCv3 draft/multiline batch instead of a teaser plus pastebin link.
        Needs the draft/multiline capability and
        supybot.protocols.irc.experimentalExtensions. 0 disables."""),
    ),
)


# ============================================================================
# Conversation Context (channel-specific with global defaults)
# ============================================================================

conf.registerChannelValue(
    LLM,
    "contextEnabled",
    registry.Boolean(
        True,
        _("""Enable conversation context (memory between messages)"""),
    ),
)

conf.registerChannelValue(
    LLM,
    "contextMaxMessages",
    registry.PositiveInteger(
        20,
        _("""Maximum messages to keep in conversation history"""),
    ),
)

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

conf.registerChannelValue(
    LLM,
    "channelContextMaxMessages",
    registry.PositiveInteger(
        10,
        _("""Maximum messages in shared channel context. This allows the bot to
        follow group conversations - when Alice asks something, Bob can continue
        the thread because the bot remembers Alice's exchange."""),
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
    "maxPromptLength",
    registry.PositiveInteger(
        10000,
        _("""Maximum length of user prompts in characters"""),
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
    "fileCleanupMax",
    registry.PositiveInteger(
        1000,
        _("""Maximum number of files to keep in HTTP directory"""),
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

conf.registerGlobalValue(
    LLM,
    "databasePath",
    registry.String(
        "",
        _("""Path to SQLite database file for persistence (reminders, usage tracking).
        If empty, uses Limnoria's data directory (data/LLM.db)."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "logLevel",
    ValidatedLogLevel(
        "WARNING",
        _("""Log level for LLM plugin (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        Set to DEBUG for verbose tracing including server response headers."""),
    ),
)

# ============================================================================
# Rate Limiting (per-command, per-tier)
#
# Tiers (checked from most to least privileged):
#   owner/admin  — always exempt (no config needed)
#   trusted      — relaxed limits (Trusted prefix)
#   registered   — standard limits (no prefix, backwards-compatible)
#   unregistered — strictest limits (Unreg prefix)
#
# Setting any count to 0 disables rate limiting for that command+tier.
# ============================================================================

conf.registerGlobalValue(
    LLM,
    "enforceRateLimits",
    registry.Boolean(
        True,
        _("""Enable per-user rate limiting for commands.
        When False, limits are tracked and logged but not enforced (monitor mode).
        Set to True to actively block requests that exceed the limit."""),
    ),
)


def _register_rate_limit_block(
    command: str,
    *,
    counts: tuple[int, int, int],  # (registered, trusted, unreg)
    windows: tuple[int, int, int],  # (registered, trusted, unreg)
) -> None:
    tiers: tuple[tuple[str, str], ...] = (
        ("", "registered tier"),
        ("Trusted", "trusted tier"),
        ("Unreg", "unregistered tier"),
    )
    for (tier, label), count, window in zip(tiers, counts, windows, strict=True):
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


# --- ask (cheapest) ---
_register_rate_limit_block(
    "ask",
    counts=(15, 15, 15),
    windows=(60, 60, 60),
)

# --- code ---
_register_rate_limit_block(
    "code",
    counts=(10, 0, 2),
    windows=(60, 60, 60),
)

# --- draw (expensive) ---
# Unregistered is the STRICTEST tier: 1 image/hour, not 0. A count of 0 means
# "rate limiting disabled" (see _is_rate_limited), so 0 would give accountless
# users UNLIMITED access to the most expensive command — the opposite of the
# intended tier ordering. Keep it a small positive cap.
_register_rate_limit_block(
    "draw",
    counts=(2, 5, 1),
    windows=(300, 60, 3600),
)

# --- meme (free: rendered locally) ---
# The ceiling is politeness to the channel, not spend.
_register_rate_limit_block(
    "meme",
    counts=(6, 12, 2),
    windows=(60, 60, 600),
)

# --- story (expensive, illustrated storybook) ---
_register_rate_limit_block(
    "story",
    counts=(2, 5, 1),
    windows=(300, 60, 3600),
)

# --- animate (most expensive: ~70s of exclusive GPU time per clip) ---
# Stricter than draw on every tier. The video box serialises work internally, so
# each accepted request delays every later one; the caps are what stops a busy
# channel from booking an hour of queue in a minute. Same reason as draw for
# keeping Unreg at 1 rather than 0 — 0 means "no limit", not "no access".
_register_rate_limit_block(
    "animate",
    counts=(2, 4, 1),
    windows=(900, 300, 7200),
)

# ============================================================================
# Assistant Tool-Calling Backend (shared by @ask, @code, @draw, invalidCommand)
# ============================================================================

conf.registerGlobalValue(
    LLM,
    "metaMaxSteps",
    registry.PositiveInteger(
        12,
        _("""Maximum tool-call round trips per assistant invocation.
        Prevents runaway tool loops."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "skipAutoWhoOnJoin",
    registry.Boolean(
        True,
        _(
            """If True (default), suppress Limnoria's automatic WHO query on channel join
            when both 'account-tag' and 'extended-join' IRCv3 capabilities are ACK'd.
            Set False to restore the legacy WHO query (emergency disable for servers
            where account-tag/extended-join misbehave). The MODE +b ban-list query is
            always suppressed regardless of this flag — nothing reads ban state."""
        ),
    ),
)

conf.registerChannelValue(
    LLM,
    "botLoopReplyLimit",
    registry.NonNegativeInteger(
        3,
        _("""How many consecutive replies the bot will give one other bot before
        going quiet. Two bots that answer each other never tire, and the network
        already knows which nicks are robots: AfterNET advertises BOT=B and marks
        them in the WHOX status field. Any human speaking in the channel clears
        the count, so a conversation a person is part of is never cut off. 0
        disables the guard entirely."""),
    ),
)

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

conf.registerChannelValue(
    LLM,
    "botLoopWindow",
    registry.PositiveInteger(
        300,
        _("""Seconds of quiet after which a bot's reply count resets. A bot that
        addresses the channel twice an hour is not a loop; this is what stops it
        being treated as one."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "maxConcurrentLLMCalls",
    registry.PositiveInteger(
        16,
        _("""Maximum number of simultaneous outbound LLM calls (across the
        command path and background work — memory extraction, watch-mode
        reminders, scheduled tasks). Lower this on small hosts or when the
        provider rate-limits aggressively."""),
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

conf.registerChannelValue(
    LLM,
    "memeModel",
    ValidatedModelName(
        "anthropic/claude-haiku-5-5",
        _("""Model that picks a template and writes captions when @meme (or
        make_meme's brief) does not name a template the catalog knows. One
        short opening list plus up to two search_templates calls per pick
        (5-19K tokens, under $0.003 on Haiku, 2-6 s).
        Empty falls back to assistantModel — but grok-4-1-fast copies the
        template's example captions instead of writing its own (measured
        2026-09-18: non-reasoning 4/4, reasoning 2/4; gemini-flash 0/4)."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "memeEditModel",
    ValidatedModelName(
        "xai/grok-imagine-image",
        _("""Image model for the picker's draw field (and make_meme's). On a
        template with picture slots it generates one picture per slot
        (xAI /v1/images/generations) that is masked into the blank; on any
        other template the captioned meme is sent to /v1/images/edits with
        the instruction. xai/ models only. About $0.02 per picture; a
        refusal is billed too."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "memeEnabled",
    registry.Boolean(
        True,
        _("""When False, @meme is refused and chat does not advertise the
        make_meme tool in this channel."""),
    ),
)

# ============================================================================
# Live IRC lookups (@channels / @names / irc_lookup tool)
# ============================================================================

conf.registerChannelValue(
    LLM,
    "ircLookupEnabled",
    registry.Boolean(
        True,
        _("""When True, chat advertises the irc_lookup tool so the model can
        answer questions about channels (LIST) and who is in one (NAMES) on
        the live network. The @channels and @names commands work regardless.
        LIST results are cached for 60 seconds per network."""),
    ),
)


conf.registerChannelValue(
    LLM,
    "reactEnabled",
    registry.Boolean(
        True,
        _("""When True, chat advertises the react tool so the model can add an
        IRCv3 emoji reaction (+draft/react) to the message it is answering or
        to another nick's latest line in the channel. A react-only turn sends
        no text. Needs the message-tags capability; without it the tool
        reports an error and the model answers in text."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "redactEnabled",
    registry.Boolean(
        True,
        _("""When True, chat advertises the delete_last_reply tool: "delete
        that" takes back the bot's latest reply in the channel with IRCv3
        REDACT (draft/message-redaction). Like react, a successful delete
        sends no text."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "notifyOnlineEnabled",
    registry.Boolean(
        True,
        _("""When True, chat advertises the notify_when_online tool: "tell me
        when Larry is on" adds Larry to the bot's IRC MONITOR list and says so
        in the channel when he signs on. Watches fire once, are kept across
        restarts, and lapse after 7 days; 5 per person."""),
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
        Time Math Utilities Seen Web Later Note Karma QuoteGrabs RSS DDG
        — write commands inside those plugins stay hidden until
        bridgeAllowMutating is True. Set this to a non-empty list to
        override the curated set with your own selection. Set
        bridgeEnabled False to disable the bridge entirely."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "bridgeAllowMutating",
    registry.Boolean(
        False,
        _("""When True, the Limnoria bridge exposes commands that modify
        persistent state (sending notes, registering feeds, mutating karma,
        etc.). When False (the default), only read-only commands are exposed
        — write commands are hidden from the LLM's tool description and any
        attempt to dispatch one returns an error envelope.

        Per-command classification lives in MUTATING_COMMANDS in
        plugins/llm/src/llm/limnoria_bridge.py."""),
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
    "bridgeScheduledTaskLimit",
    registry.NonNegativeInteger(
        5,
        _("""Maximum number of active LLM-scheduled tasks per creator in this
        channel. Enforced at create time by the schedule_llm_task tool. Set to
        0 to disable scheduling entirely.

        Each fire is checked against the user's ask rate limit but consumes no
        slot (plugin._unattended_ask_rate_limited peeks with record=False): a
        fire is skipped when the user has already maxed their own interactive
        use. This value caps the *number* of pending schedules, not their
        cumulative cost. The bridge* prefix is a leftover from the Phase 2
        bridge work that introduced scheduling; a fire runs through
        _run_unattended_assistant, which wires no bridge tools, so a scheduled
        task cannot execute Limnoria commands."""),
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

conf.registerGlobalValue(
    LLM,
    "statusQueryablePages",
    registry.SpaceSeparatedListOfStrings(
        [],
        _("""Space-separated status pages the bot can be ASKED about but never
        polls or announces, same Name=url grammar as statusPageUrls. These cost
        nothing until someone asks: they are fetched lazily and cached, with no
        incident lifecycle and no channel announcements. Use this for the long
        tail; use statusPageUrls for pages a channel should be told about. At
        most 20."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "auroraEnabled",
    registry.Boolean(
        True,
        _("""Give the chat model the aurora tool: NOAA aurora probability, solar
        wind, Kp, darkness and cloud cover for a named place. On by default."""),
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
