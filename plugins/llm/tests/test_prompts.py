"""Invariants for the consolidated prompts module.

These tests pin the shape of ``llm.prompts`` so future refactors don't
silently drop a profile key or break the shared IRC output rules.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from llm import github, prompts

_REPO_ROOT = Path(__file__).resolve().parents[3]


class TestProfilePromptInvariants:
    """Profile-facing prompts share the {bot_nick} placeholder contract."""

    @pytest.mark.parametrize(
        "name",
        ["chat", "draw", "verse", "remind_action"],
    )
    def test_profile_prompts_contain_bot_nick_placeholder(self, name):
        """Profile prompts are formatted with .format(bot_nick=...)."""
        assert "{bot_nick}" in prompts.PROMPTS[name]

    @pytest.mark.parametrize(
        "name",
        ["MEMORY_EXTRACTION_PROMPT", "MEMORY_CLEANUP_PROMPT"],
    )
    def test_memory_prompts_have_no_bot_nick_placeholder(self, name):
        """Memory prompts are used as raw constants — no .format(bot_nick=...) call.

        (They contain literal ``{"add": ...}`` JSON examples, so we can't
        assert the absence of every ``{`` — just the bot_nick contract.)
        """
        assert "{bot_nick}" not in getattr(prompts, name)


class TestIrcOutputFormatSharing:
    """IRC_OUTPUT_FORMAT is shared by Q&A modes but NOT by verse."""

    @pytest.mark.parametrize("name", ["chat", "draw", "remind_action"])
    def test_qa_prompts_embed_irc_output_format(self, name):
        """Chat-style prompts embed the shared format block verbatim."""
        assert prompts.IRC_OUTPUT_FORMAT in prompts.PROMPTS[name]

    def test_verse_does_not_embed_irc_output_format(self):
        """Verse owns its own length/format rules (no 3-line cap)."""
        assert prompts.IRC_OUTPUT_FORMAT not in prompts.PROMPTS["verse"]
        # But verse must still ban markdown in its own words — assert
        # explicitly so deleting IRC_OUTPUT_FORMAT wholesale doesn't
        # silently strip verse's markdown ban as a side effect.
        assert "Plain text only" in prompts.PROMPTS["verse"]
        assert "**bold**" in prompts.PROMPTS["verse"]


class TestProfileSpecificContent:
    """Per-profile content invariants migrated from test_assistant.py.

    Owning these here means Task 5's deletion of the equivalent block
    in test_assistant.py cannot accidentally drop a load-bearing check.
    Without these, a future edit could re-mix verse rules into the chat
    prompt or remove the tool-name anchor from code/draw and only show
    up as a behavior bug in production.
    """

    def test_chat_omits_internal_meta_token(self):
        """CHAT_SYSTEM_PROMPT does not contain the NOT_META control word."""
        assert "NOT_META" not in prompts.PROMPTS["chat"]

    def test_chat_does_not_carry_verse_rules(self):
        """Chat and verse stay structurally separate — chat must not
        mention verse-mode mechanics or it'll start cross-routing tools."""
        assert "VERSE MODE" not in prompts.PROMPTS["chat"]
        assert "verse_record" not in prompts.PROMPTS["chat"]

    def test_draw_prompt_mentions_generate_image_tool(self):
        """The draw prompt anchors the planner on the right tool name."""
        assert "generate_image" in prompts.PROMPTS["draw"]

    def test_remind_action_does_not_mention_set_reminder(self):
        """Remind-action runs INSIDE a fired reminder — telling the
        model about set_reminder there causes scheduling loops."""
        assert "set_reminder" not in prompts.PROMPTS["remind_action"]


class TestBridgeGuidanceSplit:
    """Bridge operating rules are injected per-request, not baked into chat."""

    def test_chat_prompt_does_not_mention_bridge_tools(self):
        assert "run_limnoria_command" not in prompts.CHAT_SYSTEM_PROMPT
        assert "run_limnoria_command" in prompts.BRIDGE_TOOLS_GUIDANCE


class TestOperatingGuide:
    """The chat prompt sends how-do-I-configure-you questions to the docs.

    Without it the model answered @config questions from generic Limnoria
    memory and got this bot's settings wrong.
    """

    _URLS = (prompts.LIMNORIA_GUIDE_URL, prompts.COMMANDS_GUIDE_URL)

    @pytest.mark.parametrize("url", _URLS)
    def test_chat_prompt_names_the_guide(self, url):
        assert url in prompts.CHAT_SYSTEM_PROMPT

    @pytest.mark.parametrize("url", _URLS)
    def test_guide_url_reads_verbatim_through_github(self, url):
        """A raw GitHub URL skips the search model's lossy, billed summary."""
        target = github.parse(url)
        assert target is not None
        assert target[0] == "raw"

    @pytest.mark.parametrize("url", _URLS)
    def test_guide_page_fits_the_github_clip(self, url):
        """github.digest clips at MAX_CHARS; a longer page loses its tail unseen."""
        page = _REPO_ROOT / "docs" / "guide" / url.removeprefix(prompts._DOCS_RAW + "/")
        assert page.is_file()
        header = len(f"Source: {url}\n")
        assert header + len(page.read_text(encoding="utf-8")) < github.MAX_CHARS

    def test_bridge_guidance_no_longer_says_answer_from_training(self):
        assert "training knowledge" not in prompts.BRIDGE_TOOLS_GUIDANCE


class TestIrcLookupGuidanceSplit:
    """The whois/channels HARD RULE rides only when irc_lookup is injected."""

    def test_chat_prompt_does_not_mention_irc_lookup(self):
        assert "irc_lookup" not in prompts.CHAT_SYSTEM_PROMPT
        assert "irc_lookup" in prompts.IRC_LOOKUP_GUIDANCE


class TestReactGuidanceSplit:
    """The react rule rides only when the react tool is injected."""

    def test_chat_prompt_does_not_mention_react_tool(self):
        assert "react adds" not in prompts.CHAT_SYSTEM_PROMPT
        assert "react adds" in prompts.REACT_GUIDANCE


class TestPendingTasksGuidanceSplit:
    """Reminder/scheduling rules ride only with the pending-task tools."""

    def test_chat_prompt_does_not_mention_pending_task_tools(self):
        for tool in (
            "set_reminder",
            "schedule_llm_task",
            "list_pending_tasks",
            "cancel_pending_task",
            "cancel_all_pending_tasks",
        ):
            assert tool not in prompts.CHAT_SYSTEM_PROMPT
            assert tool in prompts.PENDING_TASKS_GUIDANCE

    def test_guidance_describes_adjust_as_cancel_then_set(self):
        """No edit tool exists, so the rule must spell out the sequence."""
        guidance = prompts.PENDING_TASKS_GUIDANCE
        assert "There is no edit tool" in guidance
        rule = guidance[guidance.index("There is no edit tool") :]
        rule = rule[: rule.index("\n")]
        assert (
            rule.index("list_pending_tasks")
            < rule.index("cancel_pending_task")
            < rule.index("set_reminder")
        )

    def test_guidance_asks_before_an_implied_cancel(self):
        guidance = prompts.PENDING_TASKS_GUIDANCE
        assert "Only cancel when the user asks you to" in guidance
        assert "ask whether to cancel it" in guidance


class TestMemeGuidanceSplit:
    """The make_meme HARD RULE rides only when the tool is injected."""

    def test_chat_prompt_does_not_mention_make_meme(self):
        assert "make_meme" not in prompts.CHAT_SYSTEM_PROMPT
        assert "make_meme" in prompts.MEME_GUIDANCE
