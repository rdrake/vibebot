"""Single source of truth for per-mode chat-loop dispatch configuration.

This module owns:

- The five **chat-loop profile identifiers** (``PROFILE_CHAT``,
  ``PROFILE_CODE``, ``PROFILE_DRAW``, ``PROFILE_VERSE``,
  ``PROFILE_REMIND_ACTION``) that route requests
  through ``service.assistant_completion``.
- The :class:`Profile` frozen dataclass that bundles every per-mode fact
  for one of those routes: which Limnoria registry key holds the model,
  which ``prompts.PROMPTS`` key supplies the framework system prompt,
  which registry key holds the channel-overridable personality overlay,
  the max-output-tokens cap, and the force-search-on-explicit opt-in. The
  API key is not part of a Profile — it is resolved from the model at the
  outbound call boundary (``apikeys.api_key_for``), not carried per-route.
- The :data:`PROFILES` registry mapping each identifier to its Profile.

Tools are *not* a field on :class:`Profile`. Tool visibility is a
many-to-many relation owned by ``assistant.ToolSpec.visible_in``;
consumers should call ``assistant.get_tools_for_profile(profile.id)``
directly. This keeps a single source of truth for visibility and avoids
a back-import from ``profile`` to ``assistant``.

Lookup pattern::

    from llm.profile import PROFILES, PROFILE_CHAT
    profile = PROFILES[route_profile]  # route_profile is one of the IDs
    model_key = profile.model_setting
    prompt_text = prompts.PROMPTS[profile.prompt_id].format(bot_nick=...)
"""

from __future__ import annotations

from dataclasses import dataclass

# Route profile identifiers. These were originally declared in
# ``assistant.py``; this module is now the definition site and
# ``assistant.py`` re-imports them. The string values match the keys in
# :data:`PROFILES`, in ``prompts.PROMPTS`` for the framework prompts,
# and in the ``visible_in`` sets on ``assistant.ASSISTANT_TOOL_SPECS``.
PROFILE_CHAT = "chat"
PROFILE_CODE = "code"
PROFILE_DRAW = "draw"
PROFILE_VERSE = "verse"
PROFILE_REMIND_ACTION = "remind_action"


@dataclass(frozen=True)
class Profile:
    """Bundle of per-mode configuration for an assistant chat-loop request.

    One :class:`Profile` per ``route_profile`` string. Collapses the
    route-keyed lookups previously scattered across
    ``service.assistant_completion`` into a single named record.

    Fields:
        id: Route identifier. Must equal the dict key in :data:`PROFILES`
            and a key in ``prompts.PROMPTS``.
        model_setting: Limnoria registry key for the model name
            (e.g. ``"assistantModel"``).
        prompt_id: Key into ``prompts.PROMPTS`` for the framework system
            prompt.
        overlay_setting: Limnoria registry key for the channel-overridable
            personality overlay (e.g. ``"assistantSystemPrompt"``).
            ``None`` means this profile takes no overlay.
        max_output_tokens: ``max_tokens`` cap for the completion.
            ``None`` means unbounded.
        force_search_on_explicit: If True and an explicit search trigger
            matches the user prompt, force
            ``tool_choice={search_web}`` on step 0.
    """

    id: str
    model_setting: str
    prompt_id: str
    overlay_setting: str | None
    max_output_tokens: int | None
    force_search_on_explicit: bool


PROFILES: dict[str, Profile] = {
    PROFILE_CHAT: Profile(
        id=PROFILE_CHAT,
        model_setting="assistantModel",
        prompt_id="chat",
        overlay_setting="assistantSystemPrompt",
        # Haiku 5.5 thinks by default and thinking counts against this cap:
        # at 2000 one chat turn spent it all thinking and returned no text.
        max_output_tokens=4096,
        force_search_on_explicit=True,
    ),
    PROFILE_CODE: Profile(
        id=PROFILE_CODE,
        # The @code chat-loop planner uses assistantModel; the
        # codeModel/codeSystemPrompt registry keys belong to the *inner*
        # _code_for_assistant one-shot (plugin.py:2547), which is a tool
        # callback, not a Profile dispatch.
        model_setting="assistantModel",
        prompt_id="code",
        # The @code planner builds system_prompt from
        # user_instruction + CODE_SYSTEM_PROMPT and never reads a channel
        # overlay registry key.
        overlay_setting=None,
        max_output_tokens=None,
        force_search_on_explicit=False,
    ),
    PROFILE_DRAW: Profile(
        id=PROFILE_DRAW,
        model_setting="assistantModel",
        prompt_id="draw",
        # Same pattern as PROFILE_CODE: planner constructs its system_prompt
        # without reading a channel overlay key.
        overlay_setting=None,
        max_output_tokens=None,
        force_search_on_explicit=False,
    ),
    PROFILE_VERSE: Profile(
        id=PROFILE_VERSE,
        # Fallback only: the live verse model comes from the verseModel
        # registry key, threaded through as model_override at the dispatch
        # site (plugin.py) — same pattern as PROFILE_CODE's inner one-shot.
        model_setting="assistantModel",
        prompt_id="verse",
        overlay_setting="assistantSystemPrompt",
        # Bounded (was None/unbounded). Verse is long-form, but a
        # non-reasoning model loses coherence in the tail of a very long
        # generation — run-on sentences and gibberish — and that degraded
        # prose then poisons the next turn via self-imitation. Bumped to 3000
        # (~2250 words) on request for fuller multi-paragraph scenes; still
        # capped to bound the worst-case run-on blob, and overflow pastebins
        # via _send_long_reply.
        max_output_tokens=3000,
        force_search_on_explicit=False,
    ),
    PROFILE_REMIND_ACTION: Profile(
        id=PROFILE_REMIND_ACTION,
        model_setting="assistantModel",
        prompt_id="remind_action",
        overlay_setting="assistantSystemPrompt",
        # Same as PROFILE_CHAT: Haiku counts thinking against this cap.
        max_output_tokens=4096,
        force_search_on_explicit=True,
    ),
}
