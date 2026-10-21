"""The check_service_status tool: schema shape, visibility, and handler."""

from __future__ import annotations

import json

import pytest
from llm import assistant, statuspage
from llm.plugin import LLM
from llm.profile import PROFILE_CHAT, PROFILE_REMIND_ACTION

from .conftest import make_completion_response

CLAUDE = "https://status.claude.com"
GITHUB = "https://www.githubstatus.com"
CF = "https://www.cloudflarestatus.com"


def green_snapshot(fetched_at: float = 1000.0, *, incidents=()) -> statuspage.Snapshot:
    """Mirrors test_status_poller.py's helper of the same name."""
    return statuspage.Snapshot(
        page_name="Claude",
        page_url="https://status.claude.com",
        indicator="minor" if incidents else "none",
        description="Partial System Outage" if incidents else "All Systems Operational",
        components={"Claude API (api.anthropic.com)": "operational"},
        incidents={i.id: i for i in incidents},
        fetched_at=fetched_at,
    )


class TestToolSchema:
    def test_visible_in_chat_and_remind_action_only(self):
        spec = assistant.ASSISTANT_TOOL_REGISTRY["check_service_status"]
        assert spec.visible_in == frozenset({PROFILE_CHAT, PROFILE_REMIND_ACTION})

    def test_requires_only_llm_ask(self):
        spec = assistant.ASSISTANT_TOOL_REGISTRY["check_service_status"]
        assert spec.capability == "llm.ask"
        assert spec.require_account is False


class TestHandler:
    def _executor(self, status_fn):
        from unittest.mock import MagicMock

        return assistant.AssistantToolExecutor(
            db=MagicMock(),
            context=MagicMock(),
            nick="tester",
            channel="#test",
            status_fn=status_fn,
        )

    def test_returns_the_payload_as_json(self):
        payload = {"indicator": "none", "description": "All Systems Operational"}
        ex = self._executor(lambda **_: payload)
        assert json.loads(ex._tool_check_service_status({})) == payload

    def test_unavailable_when_not_wired(self):
        ex = self._executor(None)
        result = json.loads(ex._tool_check_service_status({}))
        assert "error" in result

    def test_callback_failure_becomes_an_error_envelope(self):
        def boom(**_):
            raise RuntimeError("no cache")

        ex = self._executor(boom)
        result = json.loads(ex._tool_check_service_status({}))
        assert "error" in result

    def test_forwards_the_service_argument(self):
        """service is now a real, schema-defined argument: it must reach the
        callback, not be dropped like an actually-unrecognised extra."""
        captured = {}

        def spy(**kwargs):
            captured.update(kwargs)
            return {"indicator": "none"}

        ex = self._executor(spy)
        ex._tool_check_service_status({"service": "anthropic"})
        assert captured["service"] == "anthropic"

    def test_blank_or_absent_service_normalizes_to_none(self):
        captured = {}

        def spy(**kwargs):
            captured.update(kwargs)
            return {"indicator": "none"}

        ex = self._executor(spy)
        ex._tool_check_service_status({"service": "   "})
        assert captured["service"] is None

        ex._tool_check_service_status({})
        assert captured["service"] is None

    def test_defaults_include_history_to_false(self):
        captured = {}

        def spy(**kwargs):
            captured.update(kwargs)
            return {"indicator": "none"}

        ex = self._executor(spy)
        ex._tool_check_service_status({})
        assert captured["include_history"] is False

    def test_passes_include_history_true_through(self):
        captured = {}

        def spy(**kwargs):
            captured.update(kwargs)
            return {"indicator": "none"}

        ex = self._executor(spy)
        ex._tool_check_service_status({"include_history": True})
        assert captured["include_history"] is True

    def _captured_include_history(self, raw):
        captured = {}

        def spy(**kwargs):
            captured.update(kwargs)
            return {"indicator": "none"}

        ex = self._executor(spy)
        ex._tool_check_service_status({"include_history": raw})
        return captured["include_history"]

    def test_stringified_true_values_parse_as_true(self):
        """Models sometimes emit stringified tool arguments (documented gap);
        bool("false") is truthy in Python, so this must be parsed, not cast."""
        for raw in ("true", "True", "TRUE", "1", "yes", "Yes"):
            assert self._captured_include_history(raw) is True, raw

    def test_stringified_false_values_parse_as_false(self):
        for raw in ("false", "False", "0", "", "no"):
            assert self._captured_include_history(raw) is False, raw

    def test_none_and_absent_parse_as_false(self):
        assert self._captured_include_history(None) is False


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


class TestToolWiringGate:
    """service.py ~5259: status_fn must be wired when EITHER statusPageUrls
    or statusQueryablePages is configured (the gate is polled OR queryable,
    per TestQueryableOnlyGate) — with both empty, status awareness is fully
    disabled, so the tool must not occupy a chat-surface slot at all."""

    def _run(
        self,
        mocker,
        make_service,
        *,
        status_page_urls: list[str],
        status_queryable_pages: list[str] | None = None,
    ):
        from llm.assistant import AssistantToolExecutor
        from llm.profile import PROFILE_CHAT

        overrides: dict[str, list[str]] = {"statusPageUrls": status_page_urls}
        if status_queryable_pages is not None:
            overrides["statusQueryablePages"] = status_queryable_pages
        service, plugin = make_service(**overrides)
        # Bind the real readers rather than stubbing them out: service.py's
        # gate must be exercised against the same registryValue fake the rest
        # of the request runs on, or a renamed/deleted registry key collapses
        # to "" here (make_registry_side_effect's fallback) while a live bot
        # raises NonExistentRegistryEntry — invisible until it hits prod.
        plugin._status_sources = LLM._status_sources.__get__(plugin)
        plugin._status_named_pages = LLM._status_named_pages.__get__(plugin)
        plugin._STATUS_MAX_SOURCES = LLM._STATUS_MAX_SOURCES
        plugin._STATUS_MAX_QUERYABLE = LLM._STATUS_MAX_QUERYABLE
        plugin._status_tool_payload = mocker.Mock(name="_status_tool_payload")
        mocker.patch(
            "llm.service.provider.chat",
            return_value=make_completion_response("hi"),
        )
        # AssistantToolExecutor is imported locally inside assistant_completion
        # (service.py ~4744), so the patch target is the defining module, not
        # llm.service's (nonexistent) module-level attribute.
        executor_spy = mocker.patch(
            "llm.assistant.AssistantToolExecutor", wraps=AssistantToolExecutor
        )

        service.assistant_completion(
            prompt="hi",
            nick="tester",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            route_profile=PROFILE_CHAT,
        )

        return executor_spy, plugin

    def test_wired_when_status_page_url_is_configured(self, mocker, make_service):
        """status_fn is now functools.partial(_status_tool_payload, pages=...)
        (the frozen-mapping binding from decision 5), so identity with the
        bare callback no longer holds — assert the partial wraps the real
        callback and freezes the resolved mapping instead."""
        executor_spy, plugin = self._run(
            mocker, make_service, status_page_urls=["https://status.claude.com"]
        )
        status_fn = executor_spy.call_args.kwargs["status_fn"]
        assert status_fn is not None
        assert status_fn.func is plugin._status_tool_payload
        assert status_fn.keywords == {"pages": {"status.claude.com": "https://status.claude.com"}}

    def test_absent_when_status_page_url_is_empty(self, mocker, make_service):
        """Neither key configured. statusQueryablePages is passed explicitly
        (rather than relying on make_registry_side_effect's unconfigured-key
        "" fallback) so the "neither" intent is legible here, not incidental."""
        executor_spy, _plugin = self._run(
            mocker, make_service, status_page_urls=[], status_queryable_pages=[]
        )
        kwargs = executor_spy.call_args.kwargs
        assert kwargs["status_fn"] is None


class TestToolSchemaGateOnConfig:
    """The schema itself, not just status_fn, must not occupy a chat-surface
    slot when the feature is unconfigured — an offered tool that can only
    answer 'not configured' still costs prompt tokens on every completion."""

    def _tool_names(
        self,
        mocker,
        make_service,
        *,
        status_page_urls: list[str],
        status_queryable_pages: list[str] | None = None,
    ) -> set[str]:
        from llm.profile import PROFILE_CHAT

        overrides: dict[str, list[str]] = {"statusPageUrls": status_page_urls}
        if status_queryable_pages is not None:
            overrides["statusQueryablePages"] = status_queryable_pages
        service, plugin = make_service(**overrides)
        # Bind the real readers rather than stubbing them out: service.py's
        # gate must be exercised against the same registryValue fake the rest
        # of the request runs on, or a renamed/deleted registry key collapses
        # to "" here (make_registry_side_effect's fallback) while a live bot
        # raises NonExistentRegistryEntry — invisible until it hits prod.
        plugin._status_sources = LLM._status_sources.__get__(plugin)
        plugin._status_named_pages = LLM._status_named_pages.__get__(plugin)
        plugin._STATUS_MAX_SOURCES = LLM._STATUS_MAX_SOURCES
        plugin._STATUS_MAX_QUERYABLE = LLM._STATUS_MAX_QUERYABLE
        plugin._status_tool_payload = mocker.Mock(name="_status_tool_payload")
        completion = mocker.patch(
            "llm.service.provider.chat",
            return_value=make_completion_response("hi"),
        )

        service.assistant_completion(
            prompt="hi",
            nick="tester",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            route_profile=PROFILE_CHAT,
        )

        tools = completion.call_args.kwargs.get("tools") or []
        return {(t.get("function", t) or {}).get("name") for t in tools}

    def test_absent_from_tool_list_when_status_page_url_is_empty(self, mocker, make_service):
        """Neither key configured, statusQueryablePages passed explicitly so
        the "neither" intent is legible rather than incidental (see
        TestToolWiringGate.test_absent_when_status_page_url_is_empty)."""
        names = self._tool_names(
            mocker, make_service, status_page_urls=[], status_queryable_pages=[]
        )
        assert "check_service_status" not in names

    def test_present_in_tool_list_when_status_page_url_is_set(self, mocker, make_service):
        names = self._tool_names(
            mocker, make_service, status_page_urls=["https://status.claude.com"]
        )
        assert "check_service_status" in names


class TestServiceEnum:
    def test_enum_lists_configured_names(self):
        from llm.assistant import get_tools_for_profile
        from llm.service import _with_status_context

        patched = _with_status_context(
            get_tools_for_profile("chat"),
            ["https://status.claude.com"],
            {"Claude": "https://status.claude.com", "CF": "https://www.cloudflarestatus.com"},
        )
        fn = next(t["function"] for t in patched if t["function"]["name"] == "check_service_status")
        assert fn["parameters"]["properties"]["service"]["enum"] == ["Claude", "CF"]

    def test_module_schema_is_not_mutated_at_any_depth(self):
        """The shipped two-level copy shares `parameters` and `properties`, so
        writing a property into them would corrupt the process-wide schema.

        The base schema (assistant.py) carries a bare `service` property with
        no `enum` — that's the static part every build starts from — so the
        risk this guards is narrower than "service" being absent entirely:
        it's the `enum` a build injects leaking into that shared object and
        surviving into the next `get_tools_for_profile` call that never asked
        for it.
        """
        from llm.assistant import get_tools_for_profile
        from llm.service import _with_status_context

        def props():
            fn = next(
                t["function"]
                for t in get_tools_for_profile("chat")
                if t["function"]["name"] == "check_service_status"
            )
            return dict(fn["parameters"]["properties"])

        before = props()
        assert "enum" not in before["service"]
        for _ in range(3):
            _with_status_context(
                get_tools_for_profile("chat"),
                ["https://status.claude.com"],
                {"Claude": "https://status.claude.com"},
            )
        assert props() == before
        assert "enum" not in props()["service"]

    def test_no_pages_omits_the_property_entirely(self):
        from llm.assistant import get_tools_for_profile
        from llm.service import _with_status_context

        patched = _with_status_context(get_tools_for_profile("chat"), [], {})
        fn = next(t["function"] for t in patched if t["function"]["name"] == "check_service_status")
        assert "service" not in fn["parameters"]["properties"]

        # The pop-branch (no pages) must not mutate the shared base schema
        # in place: a two-level copy would let props.pop("service", ...)
        # strip it from the module-level object permanently, invisible
        # unless something re-reads the base schema after this call in the
        # same process — so do that re-read here.
        base_fn = next(
            t["function"]
            for t in get_tools_for_profile("chat")
            if t["function"]["name"] == "check_service_status"
        )
        assert "service" in base_fn["parameters"]["properties"]


class TestQueryableOnlyGate:
    """The premise of the feature: polled and queryable are independent. The
    shipped gate keys on the polled list alone, so an operator with an empty
    statusPageUrls and a configured statusQueryablePages would get no tool at
    all — defeating the exact separation this feature exists to provide."""

    def test_tool_is_wired_with_no_polled_pages(self, mocker, make_service):
        from llm.assistant import AssistantToolExecutor
        from llm.profile import PROFILE_CHAT

        service, plugin = make_service(
            statusPageUrls=[], statusQueryablePages=["CF=https://www.cloudflarestatus.com"]
        )
        for attr in ("_STATUS_MAX_SOURCES", "_STATUS_MAX_QUERYABLE"):
            setattr(plugin, attr, getattr(LLM, attr))
        # Bind the REAL resolvers, not stubs: a prior review on this feature
        # found that stubbing this exact seam made the gate structurally
        # unable to detect a deleted registry key, which is how a broken
        # commit reached production.
        for meth in (
            "_status_parse_pages",
            "_status_polled_pages",
            "_status_named_pages",
            "_status_sources",
            "_status_host",
        ):
            setattr(plugin, meth, getattr(LLM, meth).__get__(plugin))
        plugin._status_tool_payload = mocker.Mock(name="_status_tool_payload")
        executor_spy = mocker.patch(
            "llm.assistant.AssistantToolExecutor", wraps=AssistantToolExecutor
        )
        completion = mocker.patch(
            "llm.service.provider.chat",
            return_value=make_completion_response("hi"),
        )

        service.assistant_completion(
            prompt="hi",
            nick="tester",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            route_profile=PROFILE_CHAT,
        )

        assert executor_spy.call_args.kwargs["status_fn"] is not None
        tools = completion.call_args.kwargs.get("tools") or []
        names = {(t.get("function", t) or {}).get("name") for t in tools}
        assert "check_service_status" in names
        # Name-only is not enough: a regression that dropped the third
        # argument to _with_status_context (e.g. passing {} instead of the
        # resolved pages) would still wire status_fn and still ship the tool
        # by name, but with `service` stripped — the model could never
        # select the queryable page, and the feature would be dead with a
        # green suite otherwise. Pin the shipped enum itself.
        fn = next(t["function"] for t in tools if t["function"]["name"] == "check_service_status")
        assert fn["parameters"]["properties"]["service"]["enum"] == ["CF"]


class TestToolPayloadOwnership:
    def test_tool_payload_does_not_advance_lifecycle_state(self, status_plugin):
        """The tool path is a NEW writer to shared state; Task 5's tests do
        not cover it."""
        from llm.plugin import LLM

        status_plugin._status_tool_payload = LLM._status_tool_payload.__get__(status_plugin)
        status_plugin._run_status_poll()  # cold start, seeds empty
        before = dict(status_plugin._status_state)  # shallow copy: catches a per-source rewrite

        status_plugin._fake_snapshot = green_snapshot(2000.0, incidents=[incident()])
        status_plugin._now = 2000.0  # clear the 30s fetch floor

        status_plugin._status_tool_payload()

        assert status_plugin._status_state == before, "the tool path must not write lifecycle state"


class TestToolPayloadStaleness:
    """The refresh/staleness branches _status_tool_payload takes around the
    read cache: cold, fresh, and stale-with-failed-refresh."""

    def test_cold_start_with_no_cache_returns_error_envelope(self, status_plugin):
        from llm.plugin import LLM

        status_plugin._status_tool_payload = LLM._status_tool_payload.__get__(status_plugin)
        status_plugin._status_read_cache = {}
        status_plugin._fake_error = statuspage.FetchError("unreachable")

        payload = status_plugin._status_tool_payload()

        service = payload["services"][0]
        assert "error" in service
        assert "indicator" not in service, "must never read as 'all systems operational'"

    def test_fresh_cache_is_served_without_a_stale_marker(self, status_plugin):
        from llm.plugin import LLM

        status_plugin._status_tool_payload = LLM._status_tool_payload.__get__(status_plugin)
        status_plugin._status_read_cache = {CLAUDE: green_snapshot(1000.0)}
        status_plugin._now = 1000.0

        payload = status_plugin._status_tool_payload()

        service = payload["services"][0]
        assert "stale" not in service
        assert "error" not in service
        assert "error" not in payload
        assert service["indicator"] == "none"

    def test_stale_cache_with_failing_refresh_is_marked_stale_and_keeps_data(self, status_plugin):
        from llm.plugin import LLM

        status_plugin._status_tool_payload = LLM._status_tool_payload.__get__(status_plugin)
        status_plugin._status_read_cache = {CLAUDE: green_snapshot(0.0)}
        status_plugin._now = status_plugin._STATUS_STALE_AFTER + 1  # past the staleness floor
        status_plugin._status_last_fetch = {}  # clear the 30s floor
        status_plugin._fake_error = statuspage.FetchError("unreachable")

        payload = status_plugin._status_tool_payload()

        service = payload["services"][0]
        assert service["stale"] is True
        assert "error" in service
        assert service["indicator"] == "none", "last-known data must survive alongside the error"


def history_entry(entry_id="inc1") -> statuspage.HistoryEntry:
    from datetime import UTC, datetime

    return statuspage.HistoryEntry(
        id=entry_id,
        name="Elevated error rates on Claude Opus 4.5",
        status="resolved",
        impact="minor",
        started_at=datetime(2026, 8, 5, 13, 55, tzinfo=UTC),
        resolved_at=datetime(2026, 8, 5, 15, 10, tzinfo=UTC),
    )


class TestStatusHistoryPayload:
    """_status_history_payload: lazy fetch, TTL cache, failure-tolerant,
    never touches _status_state or _status_read_cache."""

    def _bind(self, status_plugin):
        from llm.plugin import LLM

        status_plugin._status_history_payload = LLM._status_history_payload.__get__(status_plugin)
        return status_plugin

    def test_cache_hit_inside_ttl_does_not_refetch(self, status_plugin, mocker):
        plugin = self._bind(status_plugin)
        fetch = mocker.patch("llm.plugin.statuspage.fetch_incidents")
        plugin._status_history_cache = {CLAUDE: (history_entry(),)}
        plugin._status_history_at = {CLAUDE: 1000.0}
        plugin._now = 1000.0 + plugin._STATUS_HISTORY_TTL - 1

        result = plugin._status_history_payload(CLAUDE)

        fetch.assert_not_called()
        assert result[0]["name"] == "Elevated error rates on Claude Opus 4.5"

    def test_expiry_refetches(self, status_plugin, mocker):
        plugin = self._bind(status_plugin)
        fake_result = mocker.Mock(payload={"page": {}, "incidents": []})
        fetch = mocker.patch("llm.plugin.statuspage.fetch_incidents", return_value=fake_result)
        plugin._status_history_cache = {CLAUDE: (history_entry(),)}
        plugin._status_history_at = {CLAUDE: 1000.0}
        plugin._now = 1000.0 + plugin._STATUS_HISTORY_TTL + 1

        plugin._status_history_payload(CLAUDE)

        fetch.assert_called_once()

    def test_fetch_failure_returns_stale_cache_without_raising(self, status_plugin, mocker):
        plugin = self._bind(status_plugin)
        mocker.patch(
            "llm.plugin.statuspage.fetch_incidents", side_effect=statuspage.FetchError("down")
        )
        plugin._status_history_cache = {CLAUDE: (history_entry(),)}
        plugin._status_history_at = {CLAUDE: 0.0}
        plugin._now = 1000.0 + plugin._STATUS_HISTORY_TTL + 1

        result = plugin._status_history_payload(CLAUDE)

        assert result[0]["name"] == "Elevated error rates on Claude Opus 4.5"

    def test_invalid_payload_on_fetch_is_swallowed_like_any_other_failure(
        self, status_plugin, mocker
    ):
        plugin = self._bind(status_plugin)
        fake_result = mocker.Mock(payload={"not": "a valid history payload"})
        mocker.patch("llm.plugin.statuspage.fetch_incidents", return_value=fake_result)
        plugin._status_history_cache = {}
        plugin._status_history_at = {}
        plugin._now = 1000.0

        result = plugin._status_history_payload(CLAUDE)

        assert result == []

    def test_does_not_touch_status_state_or_read_cache(self, status_plugin, mocker):
        plugin = self._bind(status_plugin)
        fake_result = mocker.Mock(payload={"page": {}, "incidents": []})
        mocker.patch("llm.plugin.statuspage.fetch_incidents", return_value=fake_result)
        # Shallow copies, not the same object: a per-source rewrite
        # (`self._status_state[source] = ...`) replaces a value at an
        # existing key rather than replacing the dict, so an identity check
        # on the outer dict cannot see it.
        before_state = dict(plugin._status_state)
        before_read_cache = dict(plugin._status_read_cache)
        plugin._status_history_cache = {}
        plugin._status_history_at = {}
        plugin._now = 1000.0

        plugin._status_history_payload(CLAUDE)

        assert plugin._status_state == before_state
        assert plugin._status_read_cache == before_read_cache

    def test_failure_stamps_backoff_and_a_second_immediate_call_does_not_refetch(
        self, status_plugin, mocker
    ):
        """Without this, three 'when did it last go down' questions during an
        outage each pay a 30s-timeout fetch while holding an executor permit."""
        plugin = self._bind(status_plugin)
        fetch = mocker.patch(
            "llm.plugin.statuspage.fetch_incidents", side_effect=statuspage.FetchError("down")
        )
        plugin._status_history_cache = {}
        plugin._status_history_at = {}
        plugin._status_history_failed_at = {}
        plugin._now = 1000.0

        first = plugin._status_history_payload(CLAUDE)
        assert first == []
        assert plugin._status_history_failed_at[CLAUDE] == 1000.0
        fetch.assert_called_once()

        plugin._now = 1000.0 + plugin._STATUS_HISTORY_RETRY - 1
        second = plugin._status_history_payload(CLAUDE)

        # Still just the one call: the backoff window has not elapsed.
        fetch.assert_called_once()
        assert second == []

    def test_retries_after_the_backoff_window_elapses(self, status_plugin, mocker):
        plugin = self._bind(status_plugin)
        fetch = mocker.patch(
            "llm.plugin.statuspage.fetch_incidents", side_effect=statuspage.FetchError("down")
        )
        plugin._status_history_cache = {}
        plugin._status_history_at = {}
        plugin._status_history_failed_at = {}
        plugin._now = 1000.0

        plugin._status_history_payload(CLAUDE)
        fetch.assert_called_once()

        plugin._now = 1000.0 + plugin._STATUS_HISTORY_RETRY + 1
        plugin._status_history_payload(CLAUDE)

        assert fetch.call_count == 2

    def test_success_clears_the_failure_stamp(self, status_plugin, mocker):
        plugin = self._bind(status_plugin)
        fake_result = mocker.Mock(payload={"page": {}, "incidents": []})
        mocker.patch("llm.plugin.statuspage.fetch_incidents", return_value=fake_result)
        plugin._status_history_cache = {}
        plugin._status_history_at = {}
        # Past the backoff window (retry=120), or this call would itself be
        # skipped as still-backing-off and never reach the fetch.
        plugin._status_history_failed_at = {CLAUDE: 1000.0 - plugin._STATUS_HISTORY_RETRY - 1}
        plugin._now = 1000.0

        plugin._status_history_payload(CLAUDE)

        assert plugin._status_history_failed_at[CLAUDE] == 0.0


class TestStatusToolPayloadIncludeHistory:
    def test_include_history_false_has_no_recent_incidents_key(self, status_plugin):
        from llm.plugin import LLM

        status_plugin._status_tool_payload = LLM._status_tool_payload.__get__(status_plugin)
        status_plugin._status_read_cache = {CLAUDE: green_snapshot(1000.0)}
        status_plugin._now = 1000.0

        payload = status_plugin._status_tool_payload(include_history=False)

        service = payload["services"][0]
        assert "recent_incidents" not in service
        assert service["indicator"] == "none"

    def test_include_history_true_includes_recent_incidents(self, status_plugin, mocker):
        from llm.plugin import LLM

        status_plugin._status_tool_payload = LLM._status_tool_payload.__get__(status_plugin)
        status_plugin._status_history_payload = LLM._status_history_payload.__get__(status_plugin)
        status_plugin._status_read_cache = {CLAUDE: green_snapshot(1000.0)}
        status_plugin._now = 1000.0
        status_plugin._status_history_cache = {CLAUDE: (history_entry(),)}
        status_plugin._status_history_at = {CLAUDE: 1000.0}

        payload = status_plugin._status_tool_payload(include_history=True)

        service = payload["services"][0]
        assert "recent_incidents" in service
        assert service["recent_incidents"][0]["name"] == "Elevated error rates on Claude Opus 4.5"
        assert service["indicator"] == "none", "current status is always present"


class TestAggregatePayload:
    def test_every_entry_is_identified_by_configured_host(self, status_plugin):
        """page_name is third-party and absent before a first successful fetch;
        the configured host is operator truth and always present."""
        plugin = status_plugin
        plugin._registry["statusPageUrls"] = [CLAUDE, GITHUB]
        plugin._status_tool_payload = LLM._status_tool_payload.__get__(plugin)
        plugin._status_read_cache = {CLAUDE: green_snapshot(plugin._now)}

        payload = plugin._status_tool_payload()

        hosts = [e["source"] for e in payload["services"]]
        assert hosts == ["status.claude.com", "www.githubstatus.com"]

    def test_partial_failure_still_answers_for_the_healthy_source(self, status_plugin):
        plugin = status_plugin
        plugin._registry["statusPageUrls"] = [CLAUDE, GITHUB]
        plugin._status_tool_payload = LLM._status_tool_payload.__get__(plugin)
        plugin._status_read_cache = {CLAUDE: green_snapshot(plugin._now)}
        plugin._fake_error = statuspage.FetchError("unreachable")

        payload = plugin._status_tool_payload()

        claude, github = payload["services"]
        assert claude["indicator"] == "none"
        assert "error" not in claude
        assert "error" in github
        assert "error" not in payload, "a partial failure is not a tool failure"

    def test_total_failure_sets_a_top_level_error(self, status_plugin):
        """service.py:5557 treats any top-level dict without "error" as a
        successful tool call, so an all-failed services list would be recorded
        as success."""
        plugin = status_plugin
        plugin._registry["statusPageUrls"] = [CLAUDE, GITHUB]
        plugin._status_tool_payload = LLM._status_tool_payload.__get__(plugin)
        plugin._status_read_cache = {}
        plugin._status_fetch_now = lambda source, deadline=None: None

        payload = plugin._status_tool_payload()

        assert "error" in payload
        assert all("error" in e for e in payload["services"])

    def test_no_configured_sources_is_an_error(self, status_plugin):
        plugin = status_plugin
        plugin._registry["statusPageUrls"] = []
        plugin._status_tool_payload = LLM._status_tool_payload.__get__(plugin)
        assert "error" in plugin._status_tool_payload()

    def test_the_untrusted_note_appears_once_not_per_service(self, status_plugin):
        plugin = status_plugin
        plugin._registry["statusPageUrls"] = [CLAUDE, GITHUB]
        plugin._status_tool_payload = LLM._status_tool_payload.__get__(plugin)
        plugin._status_read_cache = {
            CLAUDE: green_snapshot(plugin._now),
            GITHUB: green_snapshot(plugin._now),
        }

        payload = plugin._status_tool_payload()

        assert payload["note"]
        assert all("note" not in e for e in payload["services"])

    def test_snapshot_age_is_never_negative_for_a_source_refreshed_mid_fanout(self, status_plugin):
        """A cold source's own fetch can burn real time before its snapshot
        lands. If `now` were captured once before the whole fan-out (the
        prior bug), that source's fetched_at would land after the captured
        `now` and snapshot_age_sec would go negative."""
        plugin = status_plugin
        plugin._registry["statusPageUrls"] = [CLAUDE]
        plugin._status_tool_payload = LLM._status_tool_payload.__get__(plugin)
        plugin._status_fetch_now = LLM._status_fetch_now.__get__(plugin)
        plugin._status_read_cache = {}
        plugin._status_last_fetch = {}

        real_fetch_snapshot = plugin._status_fetch_snapshot  # fixture's fake_fetch

        def slow_fetch(source, *, timeout_cap=None):
            plugin._now += 20  # the clock moves while this source's fetch is "in flight"
            return real_fetch_snapshot(source, timeout_cap=timeout_cap)

        plugin._status_fetch_snapshot = slow_fetch

        payload = plugin._status_tool_payload()

        assert payload["services"][0]["snapshot_age_sec"] >= 0


class TestToolBudget:
    def test_a_slow_source_returns_stale_rather_than_blocking(self, status_plugin):
        plugin = status_plugin
        plugin._registry["statusPageUrls"] = [CLAUDE, GITHUB]
        plugin._status_tool_payload = LLM._status_tool_payload.__get__(plugin)
        plugin._status_read_cache = {
            CLAUDE: green_snapshot(0.0),  # ancient, forces a refresh
            GITHUB: green_snapshot(0.0),
        }
        plugin._now = 100000.0

        def slow(source, *, timeout_cap=None):
            plugin._fetch_sources.append(source)
            plugin._mono += 19.0
            return green_snapshot(plugin._now)

        plugin._status_fetch_snapshot = slow
        payload = plugin._status_tool_payload()

        assert plugin._fetch_sources == [CLAUDE], "second refresh must be skipped"
        github = payload["services"][1]
        assert github["stale"] is True
        assert "error" in github

    def test_history_is_skipped_for_sources_past_the_budget(self, status_plugin, mocker):
        """223 KB per source, sequentially, inside the asking request's permit."""
        plugin = status_plugin
        plugin._registry["statusPageUrls"] = [CLAUDE, GITHUB]
        plugin._status_tool_payload = LLM._status_tool_payload.__get__(plugin)
        plugin._status_history_payload = LLM._status_history_payload.__get__(plugin)
        plugin._status_read_cache = {
            CLAUDE: green_snapshot(plugin._now),
            GITHUB: green_snapshot(plugin._now),
        }

        def slow_history(source, **kwargs):
            plugin._mono += 19.0  # burns almost the whole 20s call budget
            raise statuspage.FetchError("too slow")

        fetch = mocker.patch("llm.plugin.statuspage.fetch_incidents", side_effect=slow_history)

        payload = plugin._status_tool_payload(include_history=True)

        assert fetch.call_count == 1, "GitHub's history must not be attempted"
        assert payload["services"][0]["recent_incidents"] == []
        assert payload["services"][1]["recent_incidents"] == []


class TestDescriptionInjection:
    def test_configured_hosts_reach_the_description(self):
        from llm.assistant import get_tools_for_profile
        from llm.service import _with_status_context

        tools = get_tools_for_profile("chat")
        patched = _with_status_context(
            tools, ["https://status.claude.com", "https://www.githubstatus.com"], {}
        )
        desc = next(
            t["function"]["description"]
            for t in patched
            if t["function"]["name"] == "check_service_status"
        )
        assert "status.claude.com" in desc
        assert "www.githubstatus.com" in desc

    def test_the_shared_schema_is_never_mutated(self):
        """ToolSpec.as_tool() returns a fresh outer dict but hands back the
        SHARED module-level schema object, so an in-place edit would corrupt it
        process-wide and re-append on every completion."""
        from llm.assistant import get_tools_for_profile
        from llm.service import _with_status_context

        before = next(
            t["function"]["description"]
            for t in get_tools_for_profile("chat")
            if t["function"]["name"] == "check_service_status"
        )
        for _ in range(3):
            _with_status_context(get_tools_for_profile("chat"), ["https://status.claude.com"], {})
        after = next(
            t["function"]["description"]
            for t in get_tools_for_profile("chat")
            if t["function"]["name"] == "check_service_status"
        )
        assert before == after

    def test_other_tools_pass_through_untouched(self):
        from llm.assistant import get_tools_for_profile
        from llm.service import _with_status_context

        tools = get_tools_for_profile("chat")
        patched = _with_status_context(tools, ["https://status.claude.com"], {})
        assert len(patched) == len(tools)
        names = {t["function"]["name"] for t in patched}
        assert names == {t["function"]["name"] for t in tools}


class TestQueryCache:
    def test_first_call_fetches_and_caches(self, status_plugin):
        plugin = status_plugin
        plugin._status_query_snapshot = LLM._status_query_snapshot.__get__(plugin)
        assert plugin._status_query_snapshot(CF) is not None
        assert plugin._fetch_calls == 1
        assert CF in plugin._status_query_cache

    def test_second_call_inside_the_ttl_does_not_refetch(self, status_plugin):
        plugin = status_plugin
        plugin._status_query_snapshot = LLM._status_query_snapshot.__get__(plugin)
        plugin._status_query_snapshot(CF)
        plugin._now += plugin._STATUS_QUERY_TTL - 1
        plugin._status_query_snapshot(CF)
        assert plugin._fetch_calls == 1

    def test_past_the_ttl_refetches(self, status_plugin):
        plugin = status_plugin
        plugin._status_query_snapshot = LLM._status_query_snapshot.__get__(plugin)
        plugin._status_query_snapshot(CF)
        plugin._now += plugin._STATUS_QUERY_TTL + 1
        plugin._status_query_snapshot(CF)
        assert plugin._fetch_calls == 2

    def test_failure_is_backed_off(self, status_plugin):
        plugin = status_plugin
        plugin._status_query_snapshot = LLM._status_query_snapshot.__get__(plugin)
        plugin._fake_error = statuspage.FetchError("down")
        assert plugin._status_query_snapshot(CF) is None
        plugin._now += 1
        assert plugin._status_query_snapshot(CF) is None
        assert plugin._fetch_calls == 1, "backoff did not hold"

    def test_a_full_cycle_of_the_cap_does_not_thrash(self, status_plugin):
        """A cache smaller than the allowlist evicts every entry before it is
        reused, so every request fetches despite the TTL.

        Sized from _STATUS_MAX_QUERYABLE, not _STATUS_QUERY_CACHE_MAX: the
        failure mode this guards is cache bound < allowlist cap, and a loop
        sized from the cache's own bound can never see that gap even if the
        two constants drift apart.
        """
        plugin = status_plugin
        plugin._status_query_snapshot = LLM._status_query_snapshot.__get__(plugin)
        pages = [f"https://status{i}.example.com" for i in range(plugin._STATUS_MAX_QUERYABLE)]
        for p in pages:
            plugin._status_query_snapshot(p)
        first_pass = plugin._fetch_calls
        assert first_pass == len(pages), "first pass should be one fetch per page"
        for p in pages:
            plugin._status_query_snapshot(p)
        assert plugin._fetch_calls == first_pass, "second pass should be all cache hits"

    def test_cache_evicts_the_oldest_past_capacity(self, status_plugin):
        plugin = status_plugin
        plugin._status_query_snapshot = LLM._status_query_snapshot.__get__(plugin)
        for i in range(plugin._STATUS_QUERY_CACHE_MAX + 1):
            plugin._now += 1
            plugin._status_query_snapshot(f"https://status{i}.example.com")
        assert len(plugin._status_query_cache) == plugin._STATUS_QUERY_CACHE_MAX
        assert "https://status0.example.com" not in plugin._status_query_cache

    def test_conditional_get_uses_the_query_cache_validators(self, status_plugin, mocker):
        """_status_fetch_snapshot reads ETag from _status_read_cache, which a
        queryable page never populates — so without an explicit cached= the
        refresh is an unconditional full GET."""
        plugin = status_plugin
        plugin._status_query_snapshot = LLM._status_query_snapshot.__get__(plugin)
        plugin._status_fetch_snapshot = LLM._status_fetch_snapshot.__get__(plugin)
        fetch = mocker.patch(
            "llm.plugin.statuspage.fetch_summary",
            return_value=statuspage.FetchResult(
                payload={
                    "page": {"name": "CF", "url": CF},
                    "status": {"indicator": "none", "description": "ok"},
                    "components": [],
                    "incidents": [],
                    "scheduled_maintenances": [],
                },
                etag='W/"abc"',
                modified="Wed, 21 Oct 2015 07:28:00 GMT",
                not_modified=False,
            ),
        )
        plugin._status_query_snapshot(CF)
        plugin._now += plugin._STATUS_QUERY_TTL + 1
        plugin._status_query_snapshot(CF)
        assert fetch.call_args.kwargs["etag"] == 'W/"abc"'
        assert fetch.call_args.kwargs["modified"] == "Wed, 21 Oct 2015 07:28:00 GMT"

    def test_does_not_touch_status_state_or_read_cache(self, status_plugin):
        """The headline invariant: a queryable page must never acquire
        lifecycle state. Snapshotting with dict(...) rather than comparing to
        {} mirrors test_status_history_payload's sibling guard at :420 — a
        per-source rewrite replaces a value at an existing key, which an
        == {} check on an empty starting dict cannot distinguish from doing
        nothing."""
        plugin = status_plugin
        plugin._status_query_snapshot = LLM._status_query_snapshot.__get__(plugin)
        plugin._status_state = {CLAUDE: object()}
        plugin._status_read_cache = {CLAUDE: object()}
        before_state = dict(plugin._status_state)
        before_read_cache = dict(plugin._status_read_cache)

        plugin._status_query_snapshot(CF)

        assert plugin._status_state == before_state
        assert plugin._status_read_cache == before_read_cache

    def test_deadline_already_spent_returns_cached_without_fetching(self, status_plugin):
        plugin = status_plugin
        plugin._status_query_snapshot = LLM._status_query_snapshot.__get__(plugin)
        first = plugin._status_query_snapshot(CF)
        assert plugin._fetch_calls == 1
        plugin._now += plugin._STATUS_QUERY_TTL + 1  # past TTL: would refetch but for the deadline
        spent_deadline = plugin._status_monotonic() + plugin._STATUS_MIN_FETCH_WINDOW - 0.1
        result = plugin._status_query_snapshot(CF, deadline=spent_deadline)
        assert result is first
        assert plugin._fetch_calls == 1, "a spent deadline must not trigger a fetch"

    def test_live_deadline_reaches_the_fetch_as_timeout_cap(self, status_plugin, mocker):
        plugin = status_plugin
        plugin._status_query_snapshot = LLM._status_query_snapshot.__get__(plugin)
        snap = statuspage.Snapshot(
            page_name="CF",
            page_url=CF,
            indicator="none",
            description="ok",
            components={},
            incidents={},
            fetched_at=plugin._now,
        )
        fetch = mocker.Mock(return_value=snap)
        plugin._status_fetch_snapshot = fetch
        plugin._mono = 100.0
        deadline = 130.0

        plugin._status_query_snapshot(CF, deadline=deadline)

        assert fetch.call_args.kwargs["timeout_cap"] == pytest.approx(30.0)


class TestServiceSelection:
    def _plugin(self, status_plugin):
        p = status_plugin
        p._registry["statusPageUrls"] = ["Claude=https://status.claude.com"]
        p._registry["statusQueryablePages"] = ["CF=https://www.cloudflarestatus.com"]
        p._status_tool_payload = LLM._status_tool_payload.__get__(p)
        p._status_query_snapshot = LLM._status_query_snapshot.__get__(p)
        return p

    def test_omitted_service_returns_every_polled_source(self, status_plugin):
        plugin = self._plugin(status_plugin)
        plugin._status_read_cache = {"https://status.claude.com": green_snapshot(plugin._now)}
        payload = plugin._status_tool_payload()
        assert [e["source"] for e in payload["services"]] == ["status.claude.com"]

    def test_named_polled_page_returns_only_that_one(self, status_plugin):
        plugin = self._plugin(status_plugin)
        plugin._status_read_cache = {"https://status.claude.com": green_snapshot(plugin._now)}
        payload = plugin._status_tool_payload(service="Claude")
        assert len(payload["services"]) == 1
        assert payload["services"][0]["source"] == "status.claude.com"

    def test_named_polled_page_is_read_from_the_poller_cache_not_queried(self, status_plugin):
        """A polled source with a fresh read cache must be answered from that
        cache — not routed through the query path, which would fetch (the
        query cache starts empty) and would leave a stray entry in the query
        dicts a polled page must never touch."""
        plugin = self._plugin(status_plugin)
        plugin._status_read_cache = {"https://status.claude.com": green_snapshot(plugin._now)}
        plugin._status_tool_payload(service="Claude")
        assert plugin._fetch_calls == 0, "a fresh polled cache must not be fetched"
        assert plugin._status_query_cache == {}, "a polled source must not populate the query cache"

    def test_named_queryable_page_fetches_lazily(self, status_plugin):
        plugin = self._plugin(status_plugin)
        payload = plugin._status_tool_payload(service="CF")
        assert [e["source"] for e in payload["services"]] == ["www.cloudflarestatus.com"]
        assert plugin._fetch_calls == 1
        assert plugin._status_state == {}, "a queryable page must not gain lifecycle state"

    def test_unresolvable_service_errors_but_still_returns_polled_data(self, status_plugin):
        """service.py records any dict without "error" as a SUCCESSFUL tool
        call, so a bare note would let the model summarise unrelated healthy
        services as the requested one's state."""
        plugin = self._plugin(status_plugin)
        plugin._status_read_cache = {"https://status.claude.com": green_snapshot(plugin._now)}
        payload = plugin._status_tool_payload(service="Nope")
        assert "error" in payload
        assert "Nope" in payload["error"]
        assert payload["services"], "polled data should still ride along"

    def test_service_resolution_uses_the_frozen_mapping_when_given(self, status_plugin):
        """The enum and the dispatcher must be one snapshot; config changing
        mid-completion must not reroute an already-issued call."""
        plugin = self._plugin(status_plugin)
        frozen = {"CF": "https://www.cloudflarestatus.com"}
        plugin._registry["statusQueryablePages"] = ["CF=https://other.example.com"]
        payload = plugin._status_tool_payload(service="CF", pages=frozen)
        assert payload["services"][0]["source"] == "www.cloudflarestatus.com"

    def test_named_queryable_page_fetch_failure_sets_top_level_error(self, status_plugin):
        """service.py records any dict without "error" as a SUCCESSFUL tool
        call, so the one branch that stops the model narrating a dead
        queryable page as answered needs its own guard, not just the
        unresolvable-name branch's."""
        plugin = self._plugin(status_plugin)
        plugin._fake_error = statuspage.FetchError("boom")
        payload = plugin._status_tool_payload(service="CF")
        assert "error" in payload
        assert "error" in payload["services"][0]

    def test_named_polled_page_stale_cache_sets_top_level_error(self, status_plugin):
        plugin = self._plugin(status_plugin)
        plugin._status_read_cache = {"https://status.claude.com": green_snapshot(0.0)}
        plugin._now = plugin._STATUS_STALE_AFTER + 1  # past the staleness floor
        plugin._status_last_fetch = {}  # clear the 30s fetch floor
        plugin._fake_error = statuspage.FetchError("boom")  # refresh attempt fails

        payload = plugin._status_tool_payload(service="Claude")

        assert payload["services"][0]["stale"] is True
        assert "error" in payload

    def test_unresolvable_service_appends_to_an_unreadable_aggregate_error(self, status_plugin):
        """When every polled page is unreadable, the aggregate already set
        its own diagnostic ("No configured status page could be read.").
        `payload["services"]` is still truthy (a list of all-error entries),
        so the unresolved-name branch must append its message rather than
        overwrite — overwriting would claim "the services listed are the
        ones that are", which is false: none of them are. [Finding 3]"""
        plugin = self._plugin(status_plugin)
        plugin._fake_error = statuspage.FetchError("boom")
        payload = plugin._status_tool_payload(service="Nope")
        assert "Nope" in payload["error"]
        assert "No configured status page could be read." in payload["error"], (
            "the aggregate's own diagnostic must survive, not be overwritten"
        )
        assert "services listed are the ones that are" not in payload["error"]

    def test_unresolvable_service_with_no_polled_sources_does_not_claim_a_list(self, status_plugin):
        """statusPageUrls empty + only statusQueryablePages configured is
        reachable from Task 4 on (it gates the tool on polled OR queryable).
        The aggregate recursion then returns no "services" key at all, so the
        "the services listed are the ones that are" phrasing must not be
        glued on beside a list that doesn't exist."""
        plugin = self._plugin(status_plugin)
        plugin._registry["statusPageUrls"] = []
        payload = plugin._status_tool_payload(service="Nope")
        assert "services" not in payload
        assert "services listed" not in payload["error"]
