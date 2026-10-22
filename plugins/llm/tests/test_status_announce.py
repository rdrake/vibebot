"""Announcer: template-primary, LLM rewrite as a post-checked upgrade."""

from __future__ import annotations

import dataclasses
import json
from unittest.mock import MagicMock

import pytest
from llm import statuspage
from llm.service import LLMService

from .conftest import make_completion_response


def incident(**over) -> statuspage.IncidentView:
    base = {
        "id": "inc1",
        "name": "Elevated error rates on Claude Opus 4.5",
        "status": "investigating",
        "impact": "minor",
        "affected_components": ("Claude API (api.anthropic.com)",),
        "started_at": None,
        "created_at": None,
        "latest_update_body": "We are investigating.",
        "latest_update_at": None,
    }
    base.update(over)
    return statuspage.IncidentView(**base)


def github_snapshot(fetched_at: float = 1000.0) -> statuspage.Snapshot:
    return statuspage.Snapshot(
        page_name="GitHub",
        page_url="https://www.githubstatus.com",
        indicator="major",
        description="Incident with Actions",
        components={"Actions": "major_outage"},
        incidents={},
        fetched_at=fetched_at,
    )


def claude_snapshot(fetched_at: float = 1000.0) -> statuspage.Snapshot:
    return statuspage.Snapshot(
        page_name="Claude",
        page_url="https://status.claude.com",
        indicator="none",
        description="All Systems Operational",
        components={},
        incidents={},
        fetched_at=fetched_at,
    )


def github_incident(incident_id: str = "gh1") -> statuspage.IncidentView:
    return statuspage.IncidentView(
        id=incident_id,
        name="Incident with Actions",
        status="investigating",
        impact="major",
        affected_components=("Actions",),
        started_at=None,
        created_at=None,
        latest_update_body="We are investigating.",
        latest_update_at=None,
    )


class TestPerSourceAnnouncing:
    def test_host_check_uses_the_source_that_raised_the_incident(self, announcing_plugin):
        """allowed_host must come from the incident's own page. Deriving it from
        a single configured URL would reject every GitHub rewrite once a second
        source existed."""
        plugin = announcing_plugin
        snapshot = github_snapshot()
        plugin._status_rewrite = MagicMock(
            return_value="GitHub is having trouble — https://www.githubstatus.com/incidents/gh1"
        )
        delta = statuspage.Delta(opened=(github_incident(),))

        plugin._announce_status("https://www.githubstatus.com", delta, snapshot, lines_left=5)

        assert plugin._sent_text, "nothing was sent"
        assert "GitHub is having trouble" in plugin._sent_text[0], (
            "the rewrite must have passed _status_rewrite_ok's host check — the "
            "template fallback contains the same incident URL, so asserting on "
            "the URL alone can't tell rewrite-accepted from rewrite-rejected"
        )

    def test_marking_announced_writes_only_that_sources_state(self, announcing_plugin):
        plugin = announcing_plugin
        plugin._status_state = {}
        snapshot = github_snapshot()
        delta = statuspage.Delta(opened=(github_incident(),))

        plugin._announce_status("https://www.githubstatus.com", delta, snapshot, lines_left=5)

        assert "gh1" in plugin._status_state["https://www.githubstatus.com"].announced
        assert "https://status.claude.com" not in plugin._status_state

    def test_template_only_spends_no_completion(self, announcing_plugin):
        """Under the rewrite reserve the pass still announces — the template is
        the primary path — but must not start a completion it has no time for."""
        plugin = announcing_plugin
        plugin._status_rewrite = MagicMock(return_value="a rewrite")
        delta = statuspage.Delta(opened=(github_incident(),))

        plugin._announce_status(
            "https://www.githubstatus.com",
            delta,
            github_snapshot(),
            lines_left=5,
            template_only=True,
        )

        plugin._status_rewrite.assert_not_called()
        assert plugin._sent_text, "the template line must still go out"

    def test_uses_the_passed_snapshot_not_the_read_cache(self, announcing_plugin):
        """The read cache is writable by the tool path under 16 concurrent
        permits, so re-reading it here can label a delta with a different
        observation than the one that produced it."""
        plugin = announcing_plugin
        plugin._status_read_cache = {"https://www.githubstatus.com": claude_snapshot()}
        delta = statuspage.Delta(opened=(github_incident(),))

        plugin._announce_status(
            "https://www.githubstatus.com", delta, github_snapshot(), lines_left=5
        )

        assert "GitHub" in plugin._sent_text[0]
        assert "Claude" not in plugin._sent_text[0]


class TestGlobalLineCap:
    def test_lines_left_bounds_the_burst(self, announcing_plugin):
        plugin = announcing_plugin
        opened = tuple(github_incident(f"gh{i}") for i in range(3))
        delta = statuspage.Delta(opened=opened)

        sent = plugin._announce_status(
            "https://www.githubstatus.com", delta, github_snapshot(), lines_left=2
        )

        assert sent == 2
        assert len(plugin._sent_text) == 2

    def test_an_uncapped_incident_is_left_unmarked_for_the_next_poll(self, announcing_plugin):
        plugin = announcing_plugin
        plugin._status_state = {}
        opened = tuple(github_incident(f"gh{i}") for i in range(3))
        delta = statuspage.Delta(opened=opened)

        plugin._announce_status(
            "https://www.githubstatus.com", delta, github_snapshot(), lines_left=2
        )

        announced = plugin._status_state["https://www.githubstatus.com"].announced
        assert len(announced) == 2, "the dropped incident must stay unannounced"


class TestTemplatePath:
    def test_template_sends_when_rewrite_returns_none(self, announcing_plugin):
        plugin = announcing_plugin
        plugin._status_rewrite = MagicMock(return_value=None)
        plugin._announce_status(
            "https://status.claude.com",
            statuspage.Delta(opened=(incident(),)),
            plugin._status_read_cache["https://status.claude.com"],
            lines_left=5,
        )
        assert plugin._safe_queue.call_count == 1
        sent = plugin._sent_text[0]
        assert "Elevated error rates on Claude Opus 4.5" in sent

    def test_rewrite_is_used_when_it_passes(self, announcing_plugin):
        plugin = announcing_plugin
        plugin._status_rewrite = MagicMock(
            return_value="Heads up — Claude's API is throwing errors on Opus 4.5."
        )
        plugin._announce_status(
            "https://status.claude.com",
            statuspage.Delta(opened=(incident(),)),
            plugin._status_read_cache["https://status.claude.com"],
            lines_left=5,
        )
        assert "Heads up" in plugin._sent_text[0]

    def test_template_links_the_incident_not_the_page_root(self, announcing_plugin):
        """The RSS announcer already deep-links each incident; the status
        announcer pointed at the page root, which says nothing about which
        incident fired."""
        plugin = announcing_plugin
        plugin._status_rewrite = MagicMock(return_value=None)
        plugin._announce_status(
            "https://status.claude.com",
            statuspage.Delta(opened=(incident(id="005ym4vzrq2w"),)),
            plugin._status_read_cache["https://status.claude.com"],
            lines_left=5,
        )
        assert "https://status.claude.com/incidents/005ym4vzrq2w" in plugin._sent_text[0]

    def test_rewrite_is_handed_the_incident_permalink(self, announcing_plugin):
        plugin = announcing_plugin
        plugin._status_rewrite = MagicMock(return_value=None)
        plugin._announce_status(
            "https://status.claude.com",
            statuspage.Delta(opened=(incident(id="005ym4vzrq2w"),)),
            plugin._status_read_cache["https://status.claude.com"],
            lines_left=5,
        )
        assert (
            plugin._status_rewrite.call_args.kwargs["url"]
            == "https://status.claude.com/incidents/005ym4vzrq2w"
        )


class TestPostChecks:
    def test_rejects_rewrite_carrying_a_foreign_url(self, announcing_plugin):
        """The highest-value filter on this path: injected page text steering
        unprompted channel speech toward a link."""
        plugin = announcing_plugin
        plugin._status_rewrite = MagicMock(
            return_value="Claude is down, see https://evil.example/fix"
        )
        plugin._announce_status(
            "https://status.claude.com",
            statuspage.Delta(opened=(incident(),)),
            plugin._status_read_cache["https://status.claude.com"],
            lines_left=5,
        )
        assert "evil.example" not in plugin._sent_text[0]
        assert "Elevated error rates" in plugin._sent_text[0], "fell back to template"

    def test_accepts_a_rewrite_quoting_the_incident_permalink(self, announcing_plugin):
        """The host check is derived from the canonical source, so the deeper
        path must still pass it — otherwise every rewrite falls back to
        template."""
        plugin = announcing_plugin
        plugin._status_rewrite = MagicMock(
            return_value=(
                "Claude API is degraded — https://status.claude.com/incidents/005ym4vzrq2w"
            )
        )
        plugin._announce_status(
            "https://status.claude.com",
            statuspage.Delta(opened=(incident(id="005ym4vzrq2w"),)),
            plugin._status_read_cache["https://status.claude.com"],
            lines_left=5,
        )
        assert "/incidents/005ym4vzrq2w" in plugin._sent_text[0]

    def test_rejects_rewrite_that_never_names_the_service(self, announcing_plugin):
        plugin = announcing_plugin
        plugin._status_rewrite = MagicMock(return_value="Something somewhere is broken.")
        plugin._announce_status(
            "https://status.claude.com",
            statuspage.Delta(opened=(incident(),)),
            plugin._status_read_cache["https://status.claude.com"],
            lines_left=5,
        )
        assert "Elevated error rates" in plugin._sent_text[0]

    def test_rejects_empty_rewrite(self, announcing_plugin):
        plugin = announcing_plugin
        plugin._status_rewrite = MagicMock(return_value="   ")
        plugin._announce_status(
            "https://status.claude.com",
            statuspage.Delta(opened=(incident(),)),
            plugin._status_read_cache["https://status.claude.com"],
            lines_left=5,
        )
        assert "Elevated error rates" in plugin._sent_text[0]


class TestSendPipeline:
    def test_sanitize_output_runs_on_the_template_path(self, announcing_plugin):
        """safeArgument covers CR/LF/NUL only and explicitly not CTCP
        (plugin.py:2867-2868), and the template path carries third-party text
        nearly verbatim."""
        plugin = announcing_plugin
        plugin._status_rewrite = MagicMock(return_value=None)
        plugin._announce_status(
            "https://status.claude.com",
            statuspage.Delta(opened=(incident(),)),
            plugin._status_read_cache["https://status.claude.com"],
            lines_left=5,
        )
        assert plugin.llm_service.sanitize_output.called

    def test_multiline_rewrite_goes_out_as_one_line(self, announcing_plugin):
        """A rewrite with a newline is collapsed before the send; left raw,
        safeArgument would repr() it and the channel would see a quoted
        string with a literal \\n."""
        from llm.plugin import LLM

        plugin = announcing_plugin
        plugin._collapse_for_irc = LLM._collapse_for_irc
        plugin._status_rewrite = MagicMock(return_value="Claude is wobbly\nErrors on Opus 4.5")

        plugin._announce_status(
            "https://status.claude.com",
            statuspage.Delta(opened=(incident(),)),
            plugin._status_read_cache["https://status.claude.com"],
            lines_left=5,
        )

        assert plugin._sent_text == ["Claude is wobbly | Errors on Opus 4.5"]

    def test_line_is_truncated_at_a_word_boundary(self, announcing_plugin):
        """Chopping mid-word (or mid-URL) can change the registrable domain
        of a truncated link; truncation must fall back to the last space."""
        plugin = announcing_plugin
        full = "Claude " + ("word " * 100)
        plugin._status_rewrite = MagicMock(return_value=full)
        plugin._announce_status(
            "https://status.claude.com",
            statuspage.Delta(opened=(incident(),)),
            plugin._status_read_cache["https://status.claude.com"],
            lines_left=5,
        )
        sent = plugin._sent_text[0]
        assert len(sent) <= plugin._STATUS_ANNOUNCE_MAX_LEN
        assert sent.startswith("Claude")
        assert sent == full[: plugin._STATUS_ANNOUNCE_MAX_LEN].rsplit(" ", 1)[0]

    def test_truncation_never_splits_the_url_mid_host(self, announcing_plugin):
        plugin = announcing_plugin
        padding = "Claude " + ("detail " * 55)  # pushes the URL past the cap
        url = "https://status.claude.com/incidents/2026-08-09-elevated-errors"
        full = padding + url
        plugin._status_rewrite = MagicMock(return_value=full)
        plugin._announce_status(
            "https://status.claude.com",
            statuspage.Delta(opened=(incident(),)),
            plugin._status_read_cache["https://status.claude.com"],
            lines_left=5,
        )
        sent = plugin._sent_text[0]
        assert len(sent) <= plugin._STATUS_ANNOUNCE_MAX_LEN
        # The URL is one unbroken token with no internal space: word-boundary
        # truncation must drop it whole, never leave a partial host behind.
        assert "status.claude" not in sent or url in sent


class TestMarkingAndBudget:
    def test_marks_announced_only_on_successful_queue(self, announcing_plugin):
        plugin = announcing_plugin
        plugin._safe_queue.return_value = True
        plugin._announce_status(
            "https://status.claude.com",
            statuspage.Delta(opened=(incident(),)),
            plugin._status_read_cache["https://status.claude.com"],
            lines_left=5,
        )
        assert "inc1" in plugin._status_state["https://status.claude.com"].announced

    def test_does_not_mark_when_the_queue_drops(self, announcing_plugin):
        plugin = announcing_plugin
        plugin._safe_queue.return_value = False
        plugin._announce_status(
            "https://status.claude.com",
            statuspage.Delta(opened=(incident(),)),
            plugin._status_read_cache["https://status.claude.com"],
            lines_left=5,
        )
        state = plugin._status_state.get("https://status.claude.com", statuspage.StatusState())
        assert "inc1" not in state.announced, "must retry next poll"

    def test_empty_rendered_line_is_skipped_and_not_marked(self, announcing_plugin, mocker):
        """An empty line must never be queued, and must stay unmarked so the
        next poll retries it — otherwise a lost announcement is permanent."""
        mocker.patch("llm.plugin.statuspage.render_line", return_value="")
        announcing_plugin._status_rewrite = mocker.MagicMock(return_value=None)
        announcing_plugin._announce_status(
            "https://status.claude.com",
            statuspage.Delta(opened=(incident(),)),
            announcing_plugin._status_read_cache["https://status.claude.com"],
            lines_left=5,
        )
        assert announcing_plugin._safe_queue.call_count == 0
        state = announcing_plugin._status_state.get(
            "https://status.claude.com", statuspage.StatusState()
        )
        assert "inc1" not in state.announced

    def test_over_budget_skips_the_rewrite_but_still_announces(self, announcing_plugin):
        plugin = announcing_plugin
        plugin._status_announce_times = [plugin._now] * plugin._STATUS_ANNOUNCE_MAX_PER_HOUR
        plugin._status_rewrite = MagicMock()
        plugin._announce_status(
            "https://status.claude.com",
            statuspage.Delta(opened=(incident(),)),
            plugin._status_read_cache["https://status.claude.com"],
            lines_left=5,
        )
        assert plugin._status_rewrite.call_count == 0, "no completion when over budget"
        assert plugin._safe_queue.call_count == 1, "template still goes out"


class TestChannelSelection:
    def test_only_opted_in_channels_receive_it(self, announcing_plugin):
        plugin = announcing_plugin
        plugin._announce_channels = {"#yes": True, "#no": False}
        plugin._announce_status(
            "https://status.claude.com",
            statuspage.Delta(opened=(incident(),)),
            plugin._status_read_cache["https://status.claude.com"],
            lines_left=5,
        )
        targets = [c.args[1].args[0] for c in plugin._safe_queue.call_args_list]
        assert targets == ["#yes"]

    def test_channel_collection_is_copied_before_iteration(self, announcing_plugin):
        """Stock RSS copies (RSS/plugin.py:405); this repo iterates live and
        survives only because callers swallow. Here a RuntimeError would drop
        the outage announcement during exactly the churn an outage causes."""
        plugin = announcing_plugin
        plugin._announce_channels = {"#a": True, "#b": True}

        original = plugin._all_known_channels

        def mutating():
            result = original()
            plugin._announce_channels["#c"] = True
            return result

        plugin._all_known_channels = mutating
        plugin._announce_status(
            "https://status.claude.com",
            statuspage.Delta(opened=(incident(),)),
            plugin._status_read_cache["https://status.claude.com"],
            lines_left=5,
        )
        targets = [c.args[1].args[0] for c in plugin._safe_queue.call_args_list]
        assert "#c" not in targets, "iteration must use a snapshot, not live state"
        assert sorted(targets) == ["#a", "#b"]


class TestStatusRewrite:
    def test_completion_failure_returns_none_and_template_still_sends(self, announcing_plugin):
        """The announcer must survive a provider outage — it exists to report one."""
        from llm.plugin import LLM

        plugin = announcing_plugin
        plugin._status_rewrite = LLM._status_rewrite.__get__(plugin)
        plugin.llm_service.status_announce_completion.side_effect = RuntimeError("provider down")
        plugin._announce_status(
            "https://status.claude.com",
            statuspage.Delta(opened=(incident(),)),
            plugin._status_read_cache["https://status.claude.com"],
            lines_left=5,
        )
        assert plugin._safe_queue.call_count == 1
        assert "Elevated error rates" in plugin._sent_text[0]

    def test_rewrite_passes_only_sanitised_fields_to_the_completion(self, announcing_plugin):
        """Raw third-party prose must never reach the completion's user block."""
        from llm.plugin import LLM

        plugin = announcing_plugin
        plugin._status_rewrite = LLM._status_rewrite.__get__(plugin)
        snapshot = plugin._status_read_cache["https://status.claude.com"]
        plugin._status_rewrite(
            incident(name="bad\x01name <|tok|>"),
            "#test",
            snapshot=snapshot,
            url=snapshot.page_url,
            label="Claude",
        )
        facts = plugin.llm_service.status_announce_completion.call_args.kwargs["facts"]
        assert "\x01" not in facts["name"]
        assert "<|" not in facts["name"]

    def test_rewrite_sanitises_impact_and_affected_components(self, announcing_plugin):
        """to_tool_payload sanitises these; the rewrite path must match, not
        hand the completion two of six fields sanitised and four raw."""
        from llm.plugin import LLM

        plugin = announcing_plugin
        plugin._status_rewrite = LLM._status_rewrite.__get__(plugin)
        snapshot = plugin._status_read_cache["https://status.claude.com"]
        plugin._status_rewrite(
            incident(impact="critical\x01<|tok|>", affected_components=("API\x01x", "Code<|z|>")),
            "#test",
            snapshot=snapshot,
            url=snapshot.page_url,
            label="Claude",
        )
        facts = plugin.llm_service.status_announce_completion.call_args.kwargs["facts"]
        assert "\x01" not in facts["impact"]
        assert "<|" not in facts["impact"]
        assert all("\x01" not in c and "<|" not in c for c in facts["affected_components"])

    def test_rewrite_uses_the_caller_supplied_label_and_url_not_the_snapshot(
        self, announcing_plugin
    ):
        """label/url must come from operator config via the caller, never be
        re-read from the (third-party) snapshot page_name/page_url."""
        from llm.plugin import LLM

        plugin = announcing_plugin
        plugin._status_rewrite = LLM._status_rewrite.__get__(plugin)
        hostile_snapshot = dataclasses.replace(
            plugin._status_read_cache["https://status.claude.com"],
            page_name="Evil <|inject|> https://evil.example",
            page_url="https://evil.example",
        )
        plugin._status_rewrite(
            incident(),
            "#test",
            snapshot=hostile_snapshot,
            url="https://status.claude.com",
            label="Claude",
        )
        facts = plugin.llm_service.status_announce_completion.call_args.kwargs["facts"]
        assert facts["service"] == "Claude"
        assert facts["url"] == "https://status.claude.com"


class TestLabelSanitisation:
    """label = snapshot.page_name is third-party text (statuspage.py:215);
    it must be URL-stripped and sanitised before reaching an unprompted
    IRC line, same as any other quoted field."""

    def test_label_strips_a_url_embedded_in_the_page_name(self, announcing_plugin):
        plugin = announcing_plugin
        snapshot = dataclasses.replace(
            plugin._status_read_cache["https://status.claude.com"],
            page_name="Claude — see https://evil.example/phish for details",
        )
        plugin._status_rewrite = MagicMock(return_value=None)
        plugin._announce_status(
            "https://status.claude.com",
            statuspage.Delta(opened=(incident(),)),
            snapshot,
            lines_left=5,
        )
        sent = plugin._sent_text[0]
        assert "evil.example" not in sent
        assert "Claude" in sent

    def test_label_strips_control_tokens_from_the_page_name(self, announcing_plugin):
        plugin = announcing_plugin
        snapshot = dataclasses.replace(
            plugin._status_read_cache["https://status.claude.com"],
            page_name="Claude\x01<|inject|>",
        )
        plugin._status_rewrite = MagicMock(return_value=None)
        plugin._announce_status(
            "https://status.claude.com",
            statuspage.Delta(opened=(incident(),)),
            snapshot,
            lines_left=5,
        )
        sent = plugin._sent_text[0]
        assert "\x01" not in sent
        assert "<|" not in sent

    def test_label_falls_back_to_the_configured_host_when_page_name_is_empty(
        self, announcing_plugin
    ):
        plugin = announcing_plugin
        snapshot = dataclasses.replace(
            plugin._status_read_cache["https://status.claude.com"], page_name=""
        )
        plugin._status_rewrite = MagicMock(return_value=None)
        plugin._announce_status(
            "https://status.claude.com",
            statuspage.Delta(opened=(incident(),)),
            snapshot,
            lines_left=5,
        )
        assert "status.claude.com" in plugin._sent_text[0]


class TestSkipBeforeSpend:
    """A channel the bot has parted must be filtered out before a completion
    or an hourly-budget slot is spent on it (plugin.py ~1319-1332)."""

    def test_parted_channel_never_reaches_the_budget_or_the_rewrite(self, announcing_plugin):
        plugin = announcing_plugin
        plugin._irc_for_channel = MagicMock(return_value=None)
        plugin._status_rewrite = MagicMock()
        plugin._announce_status(
            "https://status.claude.com",
            statuspage.Delta(opened=(incident(),)),
            plugin._status_read_cache["https://status.claude.com"],
            lines_left=5,
        )
        assert plugin._status_rewrite.call_count == 0
        assert plugin._status_announce_times == []
        assert plugin._safe_queue.call_count == 0

    def test_a_live_channel_alongside_a_parted_one_still_gets_announced(self, announcing_plugin):
        plugin = announcing_plugin
        plugin._announce_channels = {"#live": True, "#gone": True}
        live_irc = MagicMock()
        plugin._irc_for_channel = MagicMock(
            side_effect=lambda ch: live_irc if ch == "#live" else None
        )
        plugin._status_rewrite = MagicMock(return_value=None)
        plugin._announce_status(
            "https://status.claude.com",
            statuspage.Delta(opened=(incident(),)),
            plugin._status_read_cache["https://status.claude.com"],
            lines_left=5,
        )
        assert plugin._safe_queue.call_count == 1
        assert plugin._status_rewrite.call_count == 1


class TestOverlaySharing:
    """The rewrite varies only with a channel's assistantSystemPrompt
    overlay, so channels sharing an overlay must share one completion —
    this both cuts cost and removes the deterministic starvation of
    alphabetically-later channels once the hourly budget runs out."""

    def test_channels_sharing_an_overlay_consume_only_one_completion(self, announcing_plugin):
        plugin = announcing_plugin
        plugin._announce_channels = {"#a": True, "#b": True, "#c": True}
        overlays = {"#a": "pirate voice", "#b": "pirate voice", "#c": "formal voice"}
        plugin.registryValue = lambda key, channel=None, *a, **k: (
            plugin._announce_channels.get(channel, False)
            if key == "statusAnnounce"
            else overlays.get(channel, "")
            if key == "assistantSystemPrompt"
            else plugin._registry.get(key)
        )
        plugin._status_rewrite = MagicMock(return_value=None)
        plugin._announce_status(
            "https://status.claude.com",
            statuspage.Delta(opened=(incident(),)),
            plugin._status_read_cache["https://status.claude.com"],
            lines_left=5,
        )
        # "#a"/"#b" share an overlay (one completion); "#c" is its own group
        # (a second completion) — three channels, two completions.
        assert plugin._status_rewrite.call_count == 2
        assert plugin._safe_queue.call_count == 3

    def test_all_deliverable_channels_in_a_shared_group_get_the_same_text(self, announcing_plugin):
        plugin = announcing_plugin
        plugin._announce_channels = {"#a": True, "#b": True}
        plugin.registryValue = lambda key, channel=None, *a, **k: (
            plugin._announce_channels.get(channel, False)
            if key == "statusAnnounce"
            else ""
            if key == "assistantSystemPrompt"
            else plugin._registry.get(key)
        )
        plugin._status_rewrite = MagicMock(return_value="Claude API is degraded right now.")
        sent = plugin._announce_status(
            "https://status.claude.com",
            statuspage.Delta(opened=(incident(),)),
            plugin._status_read_cache["https://status.claude.com"],
            lines_left=5,
        )
        assert plugin._status_rewrite.call_count == 1
        assert plugin._sent_text == [
            "Claude API is degraded right now.",
            "Claude API is degraded right now.",
        ]
        assert sent == 1, "one incident delivered to two channels counts once, not per channel"


class TestIrcForChannel:
    def test_returns_none_when_no_connection_holds_the_channel(self, announcing_plugin, mocker):
        from llm.plugin import LLM

        mocker.patch("llm.plugin.world").ircs = []
        assert LLM._irc_for_channel.__get__(announcing_plugin)("#nowhere") is None

    def test_a_connection_without_the_channel_is_not_it(self, announcing_plugin, mocker):
        """The bot parted #b: no connection is returned for it, so the
        announcer skips it before spending a completion."""
        from llm.plugin import LLM
        from supybot import ircutils

        conn = mocker.MagicMock()
        conn.state.channels = ircutils.IrcDict({"#a": object()})
        mocker.patch("llm.plugin.world").ircs = [conn]
        find = LLM._irc_for_channel.__get__(announcing_plugin)

        assert find("#b") is None
        assert find("#a") is conn

    def test_finds_a_mixed_case_channel(self, announcing_plugin, mocker):
        from llm.plugin import LLM
        from supybot import ircutils

        conn = mocker.MagicMock()
        conn.state.channels = ircutils.IrcDict({"#Linux": object()})
        mocker.patch("llm.plugin.world").ircs = [conn]
        assert LLM._irc_for_channel.__get__(announcing_plugin)("#Linux") is conn


class TestStatusAnnounceCompletion:
    """LLMService.status_announce_completion: tool-less one-shot rewrite.

    Mirrors TestParseReminderService's mock_plugin/service fixture pair
    (test_reminders.py) — the established shape for unit-testing a one-shot
    completion method with provider.chat mocked out.
    """

    @pytest.fixture
    def mock_plugin(self, mocker):
        plugin = mocker.MagicMock()
        plugin.registryValue.side_effect = lambda key, *args, **kwargs: {
            "assistantModel": "anthropic/claude-haiku-5-5",
            "timeout": 30,
            "assistantSystemPrompt": "",
        }.get(key, "")
        return plugin

    @pytest.fixture
    def service(self, mock_plugin, mocker):
        mocker.patch("llm.service.log")
        return LLMService(mock_plugin)

    def test_no_tools_key_in_optional_kwargs(self, service, mocker):
        """include_tools=False is what makes this tool-less."""
        mock_completion = mocker.patch("llm.service.provider.chat")
        mock_completion.return_value = make_completion_response("Claude API is degraded.")
        service.status_announce_completion(facts={"name": "x"}, channel="#test")
        assert "tools" not in mock_completion.call_args.kwargs

    def test_facts_arrive_as_the_json_user_block(self, service, mocker):
        mock_completion = mocker.patch("llm.service.provider.chat")
        mock_completion.return_value = make_completion_response("Claude API is degraded.")
        facts = {"name": "Elevated errors", "service": "Claude", "url": "https://status.claude.com"}
        service.status_announce_completion(facts=facts, channel="#test")
        messages = mock_completion.call_args.kwargs["messages"]
        user_msg = next(m for m in messages if m["role"] == "user")
        assert json.loads(user_msg["content"]) == facts

    def test_none_content_returns_none(self, service, mocker):
        mock_completion = mocker.patch("llm.service.provider.chat")
        mock_completion.return_value = make_completion_response(None)
        result = service.status_announce_completion(facts={"name": "x"}, channel="#test")
        assert result is None

    def test_truncated_completion_returns_none(self, service, mocker):
        """finish_reason=length means the sentence was cut off mid-thought.

        Observed in prod 2026-08-14: gemini-flash-latest spent 113 of the 120
        allotted output tokens on reasoning and emitted three, so the channel
        got the fragment "Claude incident opened" — which names the service,
        carries no URL, and therefore passed every post-check while saying
        less than the template it displaced.
        """
        mock_completion = mocker.patch("llm.service.provider.chat")
        mock_completion.return_value = make_completion_response(
            "Claude incident opened", finish_reason="length"
        )
        result = service.status_announce_completion(facts={"name": "x"}, channel="#test")
        assert result is None

    def test_complete_completion_is_returned(self, service, mocker):
        mock_completion = mocker.patch("llm.service.provider.chat")
        mock_completion.return_value = make_completion_response(
            "Claude API is degraded.", finish_reason="stop"
        )
        result = service.status_announce_completion(facts={"name": "x"}, channel="#test")
        assert result == "Claude API is degraded."

    def test_the_cap_leaves_room_for_thinking_plus_a_sentence(self, service, mocker):
        """Thinking is billed against max_tokens, so a cap sized for the
        sentence alone truncates whenever the model thinks first.
        Thinking settles around 400-550 for this prompt rather than growing
        with the cap, so the cap has to clear that, not the sentence."""
        mock_completion = mocker.patch("llm.service.provider.chat")
        mock_completion.return_value = make_completion_response("Claude API is degraded.")
        service.status_announce_completion(facts={"name": "x"}, channel="#test")
        assert mock_completion.call_args.kwargs["max_tokens"] >= 600


class TestResolutionAnnounce:
    """The all-clear rides the same delivery path as the opening it closes."""

    def test_template_says_resolved_and_links_the_incident(self, announcing_plugin):
        plugin = announcing_plugin
        plugin._status_rewrite = MagicMock(return_value=None)
        plugin._announce_status(
            "https://status.claude.com",
            statuspage.Delta(resolved=(incident(id="005ym4vzrq2w"),)),
            plugin._status_read_cache["https://status.claude.com"],
            lines_left=5,
        )
        sent = plugin._sent_text[0]
        assert "resolved" in sent
        assert "https://status.claude.com/incidents/005ym4vzrq2w" in sent

    def test_template_never_repeats_the_last_live_status(self, announcing_plugin):
        plugin = announcing_plugin
        plugin._status_rewrite = MagicMock(return_value=None)
        plugin._announce_status(
            "https://status.claude.com",
            statuspage.Delta(resolved=(incident(status="investigating"),)),
            plugin._status_read_cache["https://status.claude.com"],
            lines_left=5,
        )
        assert "investigating" not in plugin._sent_text[0]

    def test_rewrite_is_told_the_incident_is_over(self, announcing_plugin):
        plugin = announcing_plugin
        plugin._status_rewrite = MagicMock(return_value=None)
        plugin._announce_status(
            "https://status.claude.com",
            statuspage.Delta(resolved=(incident(),)),
            plugin._status_read_cache["https://status.claude.com"],
            lines_left=5,
        )
        assert plugin._status_rewrite.call_args.kwargs["event"] == "resolved"

    def test_resolution_facts_state_resolved_and_drop_the_stale_update(self, announcing_plugin):
        """A vanished incident's retained view still says 'investigating' and
        still carries its last live update body. Either one reaching the model
        invites a line that calls the outage ongoing and over at once."""
        from llm.plugin import LLM

        plugin = announcing_plugin
        plugin._status_rewrite = LLM._status_rewrite.__get__(plugin)
        plugin._status_rewrite(
            incident(),
            "#test",
            snapshot=plugin._status_read_cache["https://status.claude.com"],
            url="https://status.claude.com/incidents/inc1",
            label="Claude",
            event="resolved",
            duration_sec=3600,
        )
        facts = plugin.llm_service.status_announce_completion.call_args.kwargs["facts"]
        assert facts["event"] == "resolved"
        assert facts["status"] == "resolved"
        assert "latest_update" not in facts

    def test_resolution_facts_carry_a_human_duration_not_raw_seconds(self, announcing_plugin):
        """Observed live 2026-08-15: handed ``duration_sec=2170`` the model
        wrote "resolved after lasting 2,170 seconds", while the template it
        was upgrading says "36m" for the same incident. ``format_duration``
        already renders this for the template path; the rewrite path gets the
        same string rather than the raw integer."""
        from llm.plugin import LLM

        plugin = announcing_plugin
        plugin._status_rewrite = LLM._status_rewrite.__get__(plugin)
        plugin._status_rewrite(
            incident(),
            "#test",
            snapshot=plugin._status_read_cache["https://status.claude.com"],
            url="https://status.claude.com/incidents/inc1",
            label="Claude",
            event="resolved",
            duration_sec=3600,
        )
        facts = plugin.llm_service.status_announce_completion.call_args.kwargs["facts"]
        assert facts["duration"] == "1h"
        assert "duration_sec" not in facts, "raw seconds must not reach the model"

    def test_an_unusable_duration_is_omitted_rather_than_sent_empty(self, announcing_plugin):
        """format_duration renders "" for an undated or sub-minute incident.
        The template simply says "resolved" there, so the model is given no
        duration to narrate rather than an empty one to invent around."""
        from llm.plugin import LLM

        plugin = announcing_plugin
        plugin._status_rewrite = LLM._status_rewrite.__get__(plugin)
        plugin._status_rewrite(
            incident(),
            "#test",
            snapshot=plugin._status_read_cache["https://status.claude.com"],
            url="https://status.claude.com/incidents/inc1",
            label="Claude",
            event="resolved",
            duration_sec=None,
        )
        facts = plugin.llm_service.status_announce_completion.call_args.kwargs["facts"]
        assert "duration" not in facts
        assert "duration_sec" not in facts

    def test_opening_facts_keep_the_live_status_and_update(self, announcing_plugin):
        from llm.plugin import LLM

        plugin = announcing_plugin
        plugin._status_rewrite = LLM._status_rewrite.__get__(plugin)
        plugin._status_rewrite(
            incident(),
            "#test",
            snapshot=plugin._status_read_cache["https://status.claude.com"],
            url="https://status.claude.com/incidents/inc1",
            label="Claude",
        )
        facts = plugin.llm_service.status_announce_completion.call_args.kwargs["facts"]
        assert facts["event"] == "opened"
        assert facts["status"] == "investigating"
        assert facts["latest_update"] == "We are investigating."

    def test_marks_resolved_only_on_a_successful_queue(self, announcing_plugin):
        plugin = announcing_plugin
        plugin._safe_queue.return_value = True
        plugin._announce_status(
            "https://status.claude.com",
            statuspage.Delta(resolved=(incident(),)),
            plugin._status_read_cache["https://status.claude.com"],
            lines_left=5,
        )
        assert "inc1" in plugin._status_state["https://status.claude.com"].resolved_announced

    def test_does_not_mark_resolved_when_the_queue_drops(self, announcing_plugin):
        plugin = announcing_plugin
        plugin._safe_queue.return_value = False
        plugin._announce_status(
            "https://status.claude.com",
            statuspage.Delta(resolved=(incident(),)),
            plugin._status_read_cache["https://status.claude.com"],
            lines_left=5,
        )
        state = plugin._status_state.get("https://status.claude.com", statuspage.StatusState())
        assert "inc1" not in state.resolved_announced, "must retry next poll"

    def test_the_all_clear_does_not_mark_the_opening_announced(self, announcing_plugin):
        """Separate maps: an incident whose opening was missed must not be
        recorded as announced just because its all-clear went out."""
        plugin = announcing_plugin
        plugin._announce_status(
            "https://status.claude.com",
            statuspage.Delta(resolved=(incident(),)),
            plugin._status_read_cache["https://status.claude.com"],
            lines_left=5,
        )
        assert "inc1" not in plugin._status_state["https://status.claude.com"].announced

    def test_an_opening_and_an_all_clear_in_one_pass_both_send(self, announcing_plugin):
        plugin = announcing_plugin
        plugin._status_rewrite = MagicMock(return_value=None)
        plugin._announce_status(
            "https://status.claude.com",
            statuspage.Delta(opened=(incident(id="A"),), resolved=(incident(id="B"),)),
            plugin._status_read_cache["https://status.claude.com"],
            lines_left=5,
        )
        assert len(plugin._sent_text) == 2
        assert "(investigating)" in plugin._sent_text[0]
        assert "resolved" in plugin._sent_text[1]
