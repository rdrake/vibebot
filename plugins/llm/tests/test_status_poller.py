"""Poller lifecycle and the read-cache / lifecycle-state ownership split."""

from __future__ import annotations

import pytest
from llm import statuspage
from llm.plugin import LLM


def green_snapshot(fetched_at: float = 1000.0, *, incidents=()) -> statuspage.Snapshot:
    return statuspage.Snapshot(
        page_name="Claude",
        page_url="https://status.claude.com",
        indicator="minor" if incidents else "none",
        description="Partial System Outage" if incidents else "All Systems Operational",
        components={"Claude API (api.anthropic.com)": "operational"},
        incidents={i.id: i for i in incidents},
        fetched_at=fetched_at,
    )


def incident(incident_id="inc1") -> statuspage.IncidentView:
    return statuspage.IncidentView(
        id=incident_id,
        name="Elevated error rates on Claude Opus 4.5",
        status="investigating",
        impact="minor",
        affected_components=("Claude API (api.anthropic.com)",),
        started_at=None,
        created_at=None,
        latest_update_body="We are investigating.",
        latest_update_at=None,
    )


class TestOwnershipSplit:
    def test_inline_fetch_does_not_advance_lifecycle_state(self, status_plugin):
        """The defect two reviewers found independently: a user's query must
        not consume the announcement."""
        plugin = status_plugin
        plugin._run_status_poll()  # cold start, seeds empty
        assert plugin._status_state[CLAUDE].seeded is True

        plugin._fake_snapshot = green_snapshot(2000.0, incidents=[incident()])
        plugin._now = 2000.0  # clear the 30s fetch floor
        plugin._status_fetch_now(CLAUDE)  # the tool's inline path

        assert plugin._status_read_cache[CLAUDE].incidents, "read cache refreshed"
        assert plugin._status_state[CLAUDE].active == {}, "lifecycle state untouched"

    def test_incident_seen_first_by_the_tool_is_still_announced(self, status_plugin):
        plugin = status_plugin
        plugin._run_status_poll()
        plugin._fake_snapshot = green_snapshot(2000.0, incidents=[incident()])
        plugin._now = 2000.0  # clear the 30s fetch floor
        plugin._status_fetch_now(CLAUDE)
        plugin._run_status_poll()
        assert plugin._announce_status.call_count == 1
        delta = plugin._announce_status.call_args[0][1]
        assert [i.id for i in delta.opened] == ["inc1"]


class TestDeltaReachesTheAnnouncer:
    """The poll's own gate on calling _announce_status.

    Every other announcer test calls ``_announce_status`` directly with a
    hand-built Delta, so the branch that decides whether to call it at all was
    never covered from this side.
    """

    def test_an_all_clear_alone_still_reaches_the_announcer(self, status_plugin):
        """Observed in prod: incident hdynq1pc0fn8 cleared at 00:27:36 on
        2026-08-15 with the bot live and in the channel, and #clanker heard
        nothing. The resolved branch was added inside ``_announce_status``
        while the caller stayed gated on ``delta.opened``, so an incident that
        cleared in a pass where nothing new opened was parked in
        ``pending_resolved`` and never spoken."""
        plugin = status_plugin
        plugin._fake_snapshot = green_snapshot(1000.0, incidents=[incident()])
        plugin._run_status_poll()  # cold start seeds inc1 as live
        plugin._announce_status.reset_mock()

        plugin._fake_snapshot = green_snapshot(2000.0)  # inc1 gone from the unresolved set
        plugin._run_status_poll()

        assert plugin._announce_status.call_count == 1
        delta = plugin._announce_status.call_args[0][1]
        assert [i.id for i in delta.resolved] == ["inc1"]
        assert delta.opened == ()

    def test_a_quiet_poll_does_not_call_the_announcer(self, status_plugin):
        """The gate still has to gate: nothing opened and nothing cleared must
        not spend a pass in the announcer."""
        plugin = status_plugin
        plugin._fake_snapshot = green_snapshot(1000.0, incidents=[incident()])
        plugin._run_status_poll()
        plugin._announce_status.reset_mock()

        plugin._fake_snapshot = green_snapshot(2000.0, incidents=[incident()])
        plugin._run_status_poll()

        assert plugin._announce_status.call_count == 0


class TestFailureHandling:
    def test_fetch_error_retains_last_good_state(self, status_plugin):
        plugin = status_plugin
        plugin._fake_snapshot = green_snapshot(1000.0, incidents=[incident()])
        plugin._run_status_poll()
        before_active = dict(plugin._status_state[CLAUDE].active)

        plugin._fake_error = statuspage.FetchError("boom")
        plugin._run_status_poll()

        assert plugin._status_state[CLAUDE].active == before_active, (
            "state must not advance on failure"
        )
        assert CLAUDE in plugin._status_read_cache

    def test_invalid_payload_does_not_seed(self, status_plugin):
        plugin = status_plugin
        plugin._fake_error = statuspage.InvalidPayload("garbage")
        plugin._run_status_poll()
        assert CLAUDE not in plugin._status_state, "a bad body is not a cold start"

    def test_poll_swallows_unexpected_errors(self, status_plugin):
        plugin = status_plugin
        plugin._fake_error = RuntimeError("unexpected")
        plugin._run_status_poll()  # must not raise

    def test_fetch_error_logs_at_info_not_warning(self, status_plugin):
        """Transient and self-healing: a network blip, a timeout, a 5xx. The
        next poll retries on its own — this must not page anyone."""
        plugin = status_plugin
        plugin._fake_error = statuspage.FetchError("boom")
        plugin._run_status_poll()
        assert plugin.log.info.call_count == 1
        assert plugin.log.warning.call_count == 0

    def test_invalid_payload_logs_at_warning_not_info(self, status_plugin):
        """Structural, not transient: the page's vocabulary moved under a
        still-strict guard. A page that parsed yesterday and rejects today
        needs a human, not a silent retry — it must not share FetchError's
        log level."""
        plugin = status_plugin
        plugin._fake_error = statuspage.InvalidPayload("garbage")
        plugin._run_status_poll()
        assert plugin.log.warning.call_count == 1
        assert plugin.log.info.call_count == 0


class TestFetchFloor:
    def test_inline_fetch_respects_the_floor(self, status_plugin):
        plugin = status_plugin
        plugin._status_last_fetch = {CLAUDE: 999.0}
        plugin._now = 1000.0
        before = plugin._fetch_calls
        plugin._status_fetch_now(CLAUDE)
        assert plugin._fetch_calls == before, "inside the 30s floor, serve cache"

    def test_inline_fetch_proceeds_past_the_floor(self, status_plugin):
        plugin = status_plugin
        plugin._status_last_fetch = {CLAUDE: 900.0}
        plugin._now = 1000.0
        before = plugin._fetch_calls
        plugin._status_fetch_now(CLAUDE)
        assert plugin._fetch_calls == before + 1


class TestDisabled:
    def test_empty_url_disables_polling(self, status_plugin):
        plugin = status_plugin
        plugin._registry["statusPageUrls"] = []
        before = plugin._fetch_calls
        plugin._run_status_poll()
        assert plugin._fetch_calls == before


class TestArming:
    def test_arms_even_when_url_is_empty(self, status_plugin, mocker):
        """Re-enabling statusPageUrls must resume polling without a reload."""
        sched = mocker.patch("llm.plugin.schedule")
        status_plugin._registry["statusPageUrls"] = []
        status_plugin._schedule_status_poll()
        assert sched.addEvent.called, "an empty URL must not disarm the timer forever"

    def test_does_not_arm_when_closing(self, status_plugin, mocker):
        sched = mocker.patch("llm.plugin.schedule")
        status_plugin._llm_executor.closing = True
        status_plugin._schedule_status_poll()
        assert not sched.addEvent.called


class TestInflightDedup:
    def test_second_enqueue_while_inflight_submits_nothing(self, status_plugin, mocker):
        mocker.patch("llm.plugin.schedule")
        status_plugin._status_poll_inflight.set()
        status_plugin._enqueue_status_poll()
        assert not status_plugin._llm_executor.submit.called

    def test_idle_enqueue_submits_one_poll(self, status_plugin, mocker):
        mocker.patch("llm.plugin.schedule")

        status_plugin._enqueue_status_poll()

        assert status_plugin._llm_executor.submit.call_count == 1
        assert status_plugin._llm_executor.submit.call_args.args[0] == "status_poll"
        assert status_plugin._status_poll_inflight.is_set()

    def test_closing_enqueue_submits_nothing(self, status_plugin, mocker):
        mocker.patch("llm.plugin.schedule")
        status_plugin._llm_executor.closing = True

        status_plugin._enqueue_status_poll()

        assert not status_plugin._llm_executor.submit.called


class TestNotModified:
    def test_304_with_cache_returns_cached_snapshot_with_new_timestamp(self, status_plugin, mocker):
        cached = green_snapshot(1000.0)
        status_plugin._status_read_cache = {CLAUDE: cached}
        status_plugin._now = 2000.0
        mocker.patch(
            "llm.plugin.statuspage.fetch_summary",
            return_value=statuspage.FetchResult(
                not_modified=True, payload=None, etag='W/"a"', modified=None
            ),
        )
        snap = LLM._status_fetch_snapshot.__get__(status_plugin)(CLAUDE)
        assert snap.fetched_at == 2000.0
        assert snap.incidents == cached.incidents
        assert snap.etag == cached.etag

    def test_304_with_no_cache_raises_fetch_error(self, status_plugin, mocker):
        status_plugin._status_read_cache = {}
        mocker.patch(
            "llm.plugin.statuspage.fetch_summary",
            return_value=statuspage.FetchResult(
                not_modified=True, payload=None, etag=None, modified=None
            ),
        )
        with pytest.raises(statuspage.FetchError):
            LLM._status_fetch_snapshot.__get__(status_plugin)(CLAUDE)


class TestSourceList:
    def test_canonicalizes_dedupes_and_preserves_order(self, status_plugin):
        plugin = status_plugin
        plugin._registry["statusPageUrls"] = [
            "https://status.claude.com/",
            "https://www.githubstatus.com",
            "HTTPS://STATUS.CLAUDE.COM",
        ]
        assert plugin._status_sources() == [
            "https://status.claude.com",
            "https://www.githubstatus.com",
        ]

    def test_unusable_entries_are_dropped_not_fatal(self, status_plugin):
        plugin = status_plugin
        plugin._registry["statusPageUrls"] = [
            "not a url",
            "https://www.githubstatus.com",
        ]
        assert plugin._status_sources() == ["https://www.githubstatus.com"]

    def test_caps_at_max_sources(self, status_plugin):
        plugin = status_plugin
        plugin._registry["statusPageUrls"] = [f"https://status{i}.example.com" for i in range(9)]
        assert len(plugin._status_sources()) == plugin._STATUS_MAX_SOURCES


CLAUDE = "https://status.claude.com"
GITHUB = "https://www.githubstatus.com"
CF = "https://www.cloudflarestatus.com"


class TestPerSourceState:
    def test_pruning_clears_every_keyed_structure(self, status_plugin):
        """Pruning only _status_state would leave the other five growing without
        bound — the 5-source cap bounds the configured set, not the historical
        one."""
        plugin = status_plugin
        plugin._status_state = {GITHUB: statuspage.StatusState(seeded=True)}
        plugin._status_read_cache = {GITHUB: green_snapshot(1000.0)}
        plugin._status_last_fetch = {GITHUB: 1000.0}
        plugin._status_history_cache = {GITHUB: ()}
        plugin._status_history_at = {GITHUB: 1000.0}
        plugin._status_history_failed_at = {GITHUB: 1000.0}

        plugin._status_prune_sources([CLAUDE])

        for name in (
            "_status_state",
            "_status_read_cache",
            "_status_last_fetch",
            "_status_history_cache",
            "_status_history_at",
            "_status_history_failed_at",
        ):
            assert getattr(plugin, name) == {}, f"{name} still holds the removed source"

    def test_pruning_keeps_configured_sources(self, status_plugin):
        plugin = status_plugin
        plugin._status_state = {
            CLAUDE: statuspage.StatusState(seeded=True),
            GITHUB: statuspage.StatusState(seeded=True),
        }
        plugin._status_prune_sources([CLAUDE, GITHUB])
        assert set(plugin._status_state) == {CLAUDE, GITHUB}

    def test_fetch_floor_is_per_source(self, status_plugin):
        """One source's inline fetch must not suppress another's."""
        plugin = status_plugin
        plugin._now = 1000.0
        plugin._status_last_fetch = {CLAUDE: 1000.0}
        plugin._status_fetch_now(GITHUB)
        assert plugin._fetch_calls == 1, "GitHub blocked by Claude's floor"
        plugin._status_fetch_now(CLAUDE)
        assert plugin._fetch_calls == 1, "Claude's own floor did not hold"


class TestPassDeadline:
    def test_a_slow_source_defers_the_rest_and_sets_the_cursor(self, status_plugin):
        """A budget checked only between sources bounds nothing; the deferred
        source must be where the next pass starts."""
        plugin = status_plugin
        plugin._registry["statusPageUrls"] = [CLAUDE, GITHUB]

        def slow_fetch(source, *, timeout_cap=None):
            plugin._fetch_sources.append(source)
            plugin._mono += 44.0  # burn nearly the whole 45s budget
            return green_snapshot(plugin._now)

        plugin._status_fetch_snapshot = slow_fetch
        plugin._run_status_poll()

        assert plugin._fetch_sources == [CLAUDE], "second source should be deferred"
        assert plugin._status_cursor == GITHUB, "next pass must resume at GitHub"

    def test_next_pass_starts_at_the_cursor(self, status_plugin):
        plugin = status_plugin
        plugin._registry["statusPageUrls"] = [CLAUDE, GITHUB]
        plugin._status_cursor = GITHUB
        plugin._run_status_poll()
        assert plugin._fetch_sources == [GITHUB, CLAUDE]

    def test_a_completed_pass_clears_the_cursor(self, status_plugin):
        plugin = status_plugin
        plugin._registry["statusPageUrls"] = [CLAUDE, GITHUB]
        plugin._run_status_poll()
        assert plugin._status_cursor is None

    def test_a_cursor_no_longer_configured_restarts_at_the_head(self, status_plugin):
        plugin = status_plugin
        plugin._registry["statusPageUrls"] = [CLAUDE]
        plugin._status_cursor = "https://gone.example.com"
        plugin._run_status_poll()
        assert plugin._fetch_sources == [CLAUDE]

    def test_a_failing_source_does_not_pin_the_head_of_the_rotation(self, status_plugin):
        """Advancing the cursor only on success would let one broken page
        starve every source behind it, forever."""
        plugin = status_plugin
        plugin._registry["statusPageUrls"] = [CLAUDE, GITHUB]

        def failing_first(source, *, timeout_cap=None):
            plugin._fetch_sources.append(source)
            if source == CLAUDE:
                raise statuspage.FetchError("boom")
            return green_snapshot(plugin._now)

        plugin._status_fetch_snapshot = failing_first
        plugin._run_status_poll()
        assert plugin._fetch_sources == [CLAUDE, GITHUB], "GitHub must still be polled"
        assert plugin._status_cursor is None


class TestSourceIsolation:
    def test_one_dead_source_does_not_stop_the_others(self, status_plugin):
        plugin = status_plugin
        plugin._registry["statusPageUrls"] = [CLAUDE, GITHUB]

        def half_broken(source, *, timeout_cap=None):
            plugin._fetch_sources.append(source)
            if source == CLAUDE:
                raise statuspage.FetchError("unreachable")
            return green_snapshot(plugin._now, incidents=[incident()])

        plugin._status_fetch_snapshot = half_broken
        plugin._run_status_poll()

        assert CLAUDE not in plugin._status_read_cache
        assert GITHUB in plugin._status_read_cache
        assert plugin._status_state[GITHUB].seeded is True

    def test_state_does_not_leak_between_sources(self, status_plugin):
        plugin = status_plugin
        plugin._registry["statusPageUrls"] = [CLAUDE, GITHUB]
        plugin._run_status_poll()  # cold start seeds both empty

        def github_only_incident(source, *, timeout_cap=None):
            plugin._fetch_sources.append(source)
            if source == GITHUB:
                return green_snapshot(plugin._now, incidents=[incident("gh1")])
            return green_snapshot(plugin._now)

        plugin._status_fetch_snapshot = github_only_incident
        plugin._run_status_poll()

        assert "gh1" in plugin._status_state[GITHUB].active
        assert plugin._status_state[CLAUDE].active == {}


class TestShutdownDuringAPass:
    def test_unload_stops_the_pass_before_the_next_source(self, status_plugin):
        """die() waits only 2s for running jobs, and the poll does not check
        closing today — a multi-source pass would keep fetching and running
        billed rewrites long after unload."""
        plugin = status_plugin
        plugin._registry["statusPageUrls"] = [CLAUDE, GITHUB]

        def close_after_first(source, *, timeout_cap=None):
            plugin._fetch_sources.append(source)
            plugin._llm_executor.closing = True
            return green_snapshot(plugin._now)

        plugin._status_fetch_snapshot = close_after_first
        plugin._run_status_poll()
        assert plugin._fetch_sources == [CLAUDE]


class TestDeadlinePropagation:
    """The budget must reach the work, not just gate the loop between sources."""

    def test_timeout_cap_is_bounded_and_shrinks_across_sources(self, status_plugin):
        plugin = status_plugin
        plugin._registry["statusPageUrls"] = [CLAUDE, GITHUB]
        caps = []

        def recording_fetch(source, *, timeout_cap=None):
            plugin._fetch_sources.append(source)
            caps.append(timeout_cap)
            plugin._mono += 10.0
            return green_snapshot(plugin._now)

        plugin._status_fetch_snapshot = recording_fetch
        plugin._run_status_poll()

        assert len(caps) == 2
        assert caps[0] <= plugin._STATUS_PASS_BUDGET, (
            "the first cap must not exceed the pass budget"
        )
        assert caps[1] < caps[0], "the second source's cap must reflect what the first spent"

    def test_low_remaining_budget_forces_template_only(self, status_plugin):
        plugin = status_plugin
        plugin._registry["statusPageUrls"] = [CLAUDE, GITHUB]
        plugin._run_status_poll()  # cold start seeds both sources quiet

        def burn_then_incident(source, *, timeout_cap=None):
            plugin._fetch_sources.append(source)
            if source == CLAUDE:
                plugin._mono += 30.0  # 45s budget - 30s = 15s left, under the 20s reserve
                return green_snapshot(plugin._now)
            return green_snapshot(plugin._now, incidents=[incident()])

        plugin._status_fetch_snapshot = burn_then_incident
        plugin._run_status_poll()

        assert plugin._announce_status.call_args.kwargs["template_only"] is True

    def test_timeout_cap_bounds_the_real_fetch_call(self, status_plugin, mocker):
        """The caller-computed cap is worthless unless the callee's socket
        timeout actually shrinks to match it. This binds the real
        _status_fetch_snapshot (not the fake_fetch fixture stand-in every
        other poller/tool test uses) and asserts what reaches
        statuspage.fetch_summary."""
        plugin = status_plugin
        plugin._registry["timeout"] = 30  # above the cap, so the cap must be what wins
        cached = green_snapshot(1000.0)
        plugin._status_read_cache = {CLAUDE: cached}
        mock_fetch = mocker.patch(
            "llm.plugin.statuspage.fetch_summary",
            return_value=statuspage.FetchResult(
                not_modified=True, payload=None, etag=None, modified=None
            ),
        )
        timeout_cap = 5.0  # below both the registry timeout and the 30s ceiling

        LLM._status_fetch_snapshot.__get__(plugin)(CLAUDE, timeout_cap=timeout_cap)

        assert mock_fetch.call_args.kwargs["timeout"] <= timeout_cap


class TestAnnouncerIsReachedPerSource:
    def test_each_source_announces_its_own_incident(self, status_plugin):
        plugin = status_plugin
        plugin._registry["statusPageUrls"] = [CLAUDE, GITHUB]
        plugin._run_status_poll()  # cold start seeds both

        def both_broken(source, *, timeout_cap=None):
            plugin._fetch_sources.append(source)
            name = "c1" if source == CLAUDE else "g1"
            return green_snapshot(plugin._now, incidents=[incident(name)])

        plugin._status_fetch_snapshot = both_broken
        plugin._announce_status.reset_mock()
        plugin._announce_status.return_value = 1
        plugin._run_status_poll()

        announced = {
            call.args[0]: [i.id for i in call.args[1].opened]
            for call in plugin._announce_status.call_args_list
        }
        assert announced == {CLAUDE: ["c1"], GITHUB: ["g1"]}


THIRD = "https://status.example.com"


class TestGlobalLineBudget:
    """_STATUS_MAX_LINES_PER_POLL is only enforced by _run_status_poll on the
    caller side of the announcer seam (lines_left -= self._poll_one_source(...)).
    Nothing on the callee side proves the budget actually shrinks across
    sources or stops a source once it is spent."""

    def test_second_source_receives_the_shrunk_budget(self, status_plugin):
        plugin = status_plugin
        plugin._registry["statusPageUrls"] = [CLAUDE, GITHUB]
        plugin._run_status_poll()  # cold start seeds both, quiet

        def both_incident(source, *, timeout_cap=None):
            plugin._fetch_sources.append(source)
            name = "c1" if source == CLAUDE else "g1"
            return green_snapshot(plugin._now, incidents=[incident(name)])

        plugin._status_fetch_snapshot = both_incident
        plugin._announce_status.reset_mock()
        plugin._announce_status.return_value = 4  # _STATUS_MAX_LINES_PER_POLL (5) - 4 = 1 left
        plugin._run_status_poll()

        assert plugin._announce_status.call_count == 2
        assert plugin._announce_status.call_args_list[1].kwargs["lines_left"] == 1

    def test_a_third_source_is_never_announced_once_the_budget_is_spent(self, status_plugin):
        plugin = status_plugin
        plugin._registry["statusPageUrls"] = [CLAUDE, GITHUB, THIRD]
        plugin._run_status_poll()  # cold start seeds all three, quiet

        def all_incidents(source, *, timeout_cap=None):
            plugin._fetch_sources.append(source)
            name = {CLAUDE: "c1", GITHUB: "g1", THIRD: "t1"}[source]
            return green_snapshot(plugin._now, incidents=[incident(name)])

        plugin._status_fetch_snapshot = all_incidents
        plugin._announce_status.reset_mock()
        plugin._announce_status.return_value = 4  # first source alone exhausts the budget
        plugin._run_status_poll()

        assert plugin._announce_status.call_count == 2, (
            "the third source must not be announced once lines_left is spent"
        )
        announced_sources = {call.args[0] for call in plugin._announce_status.call_args_list}
        assert THIRD not in announced_sources


class TestPageGrammar:
    """A bare URL stays valid and takes its host as its name, which is what
    statusPageUrls entries did before names existed."""

    def test_named_and_bare_entries_both_parse(self, status_plugin):
        plugin = status_plugin
        plugin._registry["statusPageUrls"] = [
            "Claude=https://status.claude.com",
            "https://www.githubstatus.com",
        ]
        assert plugin._status_named_pages() == {
            "Claude": "https://status.claude.com",
            "www.githubstatus.com": "https://www.githubstatus.com",
        }

    @pytest.mark.parametrize(
        "entry",
        [
            "=https://status.claude.com",  # empty name
            "has space=https://x.example",  # impossible in a space list, but explicit
            "toolongname" * 5 + "=https://x.example",
            "bad!name=https://x.example",
            "Name=not a url",
            "Name=ftp://x.example",
            "Name=https://x.example/path",
        ],
    )
    def test_unusable_entries_are_dropped_not_fatal(self, status_plugin, entry):
        plugin = status_plugin
        plugin._registry["statusPageUrls"] = [entry, "Good=https://status.claude.com"]
        assert plugin._status_named_pages() == {"Good": "https://status.claude.com"}

    def test_duplicate_name_keeps_the_first(self, status_plugin):
        plugin = status_plugin
        plugin._registry["statusPageUrls"] = [
            "Dup=https://status.claude.com",
            "dup=https://www.githubstatus.com",
        ]
        assert plugin._status_named_pages() == {"Dup": "https://status.claude.com"}

    def test_two_names_one_canonical_source_drops_the_later(self, status_plugin):
        """A silent skip would show the operator a valid-looking entry in
        @config that never appears in the enum."""
        plugin = status_plugin
        plugin._registry["statusPageUrls"] = [
            "Foo=https://status.claude.com",
            "Bar=https://status.claude.com/",
        ]
        assert plugin._status_named_pages() == {"Foo": "https://status.claude.com"}

    def test_long_bare_host_is_still_polled(self, status_plugin):
        """A bare URL's derived name used to be checked against
        _STATUS_PAGE_NAME_RE unmodified: a >32-char host (some AWS regional
        status hosts are exactly this shape) hit `continue` before the page
        was ever added, so it silently stopped being polled. [Finding 1]"""
        plugin = status_plugin
        long_host_url = "https://status.us-gov-west-1.amazonaws.com"
        assert len(plugin._status_host(long_host_url)) > 32, "fixture must exercise the >32 case"
        plugin._registry["statusPageUrls"] = [long_host_url]
        named = plugin._status_named_pages()
        assert list(named.values()) == [long_host_url], (
            "the long-host bare URL must still be polled"
        )

    def test_bare_urls_differing_only_by_port_both_survive(self, status_plugin):
        """Two distinct canonical sources (different ports) used to derive
        the same name from urlparse(...).hostname alone, so the second was
        dropped as a duplicate name even though both were configured.
        [Finding 1]"""
        plugin = status_plugin
        plugin._registry["statusPageUrls"] = ["https://x.example", "https://x.example:8443"]
        named = plugin._status_named_pages()
        assert len(named) == 2, "both ports must survive as distinct pages"
        assert len(set(named)) == 2, "the derived names must be distinct"
        assert set(named.values()) == {"https://x.example", "https://x.example:8443"}


class TestEnumIsConfigOnly:
    """The central design invariant: selector names come from operator config
    alone, never from a fetched payload. The design's first version was
    rejected specifically for deriving the enum from a snapshot's page_name
    — a compromised or renamed page could otherwise capture another page's
    selector. Nothing pinned this before; a regression seeding names from
    the read cache turned no test red. [Finding 2]"""

    def test_named_pages_ignores_fetched_page_name_even_after_a_poll(self, status_plugin):
        plugin = status_plugin
        plugin._registry["statusPageUrls"] = [
            "Claude=https://status.claude.com",
            "CF=https://www.cloudflarestatus.com",
        ]

        before = plugin._status_named_pages()
        assert before == {"Claude": CLAUDE, "CF": CF}

        # The polled page's own page_name — untrusted, third-party — happens
        # to equal the OTHER configured page's operator-chosen name. This is
        # the exact shape an attacker or a careless page rename could
        # produce. Setting _fake_snapshot (not just pre-seeding the cache
        # directly) matters: _run_status_poll's own fetch would otherwise
        # overwrite a pre-seeded entry with an unpoisoned one before the
        # assertion below ever sees it.
        spoofed = statuspage.Snapshot(
            page_name="CF",
            page_url=CLAUDE,
            indicator="none",
            description="All Systems Operational",
            components={},
            incidents={},
            fetched_at=plugin._now,
        )
        plugin._fake_snapshot = spoofed

        plugin._run_status_poll()

        assert plugin._status_read_cache[CLAUDE].page_name == "CF", (
            "fixture must actually poison the read cache"
        )

        after = plugin._status_named_pages()
        assert after == {"Claude": CLAUDE, "CF": CF}, (
            "a fetched page_name must never capture another page's configured selector"
        )
        assert list(after) == list(before), "poll-driven cache fills must not reorder the enum"
