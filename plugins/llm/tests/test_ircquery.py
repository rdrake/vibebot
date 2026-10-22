"""LIST/NAMES numeric correlation and the one-line formatters behind them."""

from __future__ import annotations

import threading
from collections.abc import Callable

from llm import ircquery
from llm.ircquery import ChannelRow, IrcQueryRegistry


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def rows(*specs: tuple[str, int, str]) -> list[ChannelRow]:
    return [ChannelRow(name=n, users=u, topic=t) for n, u, t in specs]


def ask_twice(ask: Callable[[Callable[[], None]], object], answer: Callable[[], None]) -> tuple:
    """Run ``ask`` on two threads during one in-flight query, then ``answer`` it.

    The second asker starts only after the first has sent, so it finds the
    query in flight. Returns (number of sends, results in finishing order).
    """
    sends: list[int] = []
    first_sent = threading.Event()
    results: list[object] = []

    def send() -> None:
        sends.append(1)
        first_sent.set()

    t1 = threading.Thread(target=lambda: results.append(ask(send)))
    t1.start()
    assert first_sent.wait(1.0)
    t2 = threading.Thread(target=lambda: results.append(ask(send)))
    t2.start()
    # Give t2 a moment to attach to the in-flight query before the answer.
    t2.join(0.05)
    answer()
    t1.join(3.0)
    t2.join(3.0)
    return len(sends), results


class TestListChannels:
    def test_returns_rows_fed_between_send_and_end(self):
        """Rows arriving after LIST is sent are returned once 323 closes it."""
        reg = IrcQueryRegistry()

        def send() -> None:
            reg.on_list_row("afternet", "#a", 3, "alpha")
            reg.on_list_row("afternet", "#b", 12, "")
            reg.on_list_end("afternet")

        result = reg.list_channels("afternet", send, timeout=1.0)

        assert result == rows(("#a", 3, "alpha"), ("#b", 12, ""))

    def test_timeout_returns_none_and_clears_the_pending_query(self):
        """A server that never sends 323 must not wedge every later LIST."""
        reg = IrcQueryRegistry()
        sends: list[int] = []

        assert reg.list_channels("afternet", lambda: sends.append(1), timeout=0.01) is None
        assert reg.list_channels("afternet", lambda: sends.append(1), timeout=0.01) is None
        assert len(sends) == 2

    def test_fresh_result_is_served_from_cache_without_resending(self):
        """LIST is expensive server-side; within the TTL nobody re-sends it."""
        clock = FakeClock()
        reg = IrcQueryRegistry(clock=clock, cache_ttl=60.0)
        sends: list[int] = []

        def send() -> None:
            sends.append(1)
            reg.on_list_row("afternet", "#a", 1, "")
            reg.on_list_end("afternet")

        first = reg.list_channels("afternet", send, timeout=1.0)
        clock.now += 30
        second = reg.list_channels("afternet", send, timeout=1.0)
        clock.now += 31
        third = reg.list_channels("afternet", send, timeout=1.0)

        assert first == second == third
        assert len(sends) == 2

    def test_cache_is_per_network(self):
        reg = IrcQueryRegistry(cache_ttl=60.0)
        sends: list[str] = []

        def sender(network: str):
            def send() -> None:
                sends.append(network)
                reg.on_list_row(network, f"#{network}", 1, "")
                reg.on_list_end(network)

            return send

        reg.list_channels("a", sender("a"), timeout=1.0)
        reg.list_channels("b", sender("b"), timeout=1.0)

        assert sends == ["a", "b"]

    def test_concurrent_callers_share_one_in_flight_list(self):
        """Two askers during the same LIST get the same rows from one request."""
        reg = IrcQueryRegistry()
        sends: list[int] = []
        first_sent = threading.Event()
        results: list[list[ChannelRow] | None] = []

        def send() -> None:
            sends.append(1)
            first_sent.set()

        def ask() -> None:
            results.append(reg.list_channels("afternet", send, timeout=2.0))

        t1 = threading.Thread(target=ask)
        t1.start()
        assert first_sent.wait(1.0)
        t2 = threading.Thread(target=ask)
        t2.start()
        # Give t2 a moment to attach to the in-flight query before the end.
        t2.join(0.05)
        reg.on_list_row("afternet", "#x", 5, "")
        reg.on_list_end("afternet")
        t1.join(2.0)
        t2.join(2.0)

        assert len(sends) == 1
        assert results == [rows(("#x", 5, "")), rows(("#x", 5, ""))]

    def test_rows_without_a_pending_query_are_ignored(self):
        """A stray 322 (say, from a LIST the operator typed) must not crash."""
        reg = IrcQueryRegistry()
        reg.on_list_row("afternet", "#a", 1, "")
        reg.on_list_end("afternet")

    def test_server_error_ends_the_query_with_that_message(self):
        """263 RPL_TRYAGAIN for LIST surfaces as an error, not a timeout."""
        reg = IrcQueryRegistry()

        def send() -> None:
            reg.on_error("afternet", "LIST", "Please wait a while and try again.")

        result = reg.list_channels("afternet", send, timeout=1.0)

        assert result is None
        assert reg.last_error("afternet", "LIST") == "Please wait a while and try again."

    def test_errors_nobody_waited_for_are_not_remembered(self):
        """Every 401 on the network passes through on_error; only pending ones stick."""
        reg = IrcQueryRegistry()
        reg.on_error("afternet", "nosuchnick", "No such nick/channel")
        assert reg.last_error("afternet", "nosuchnick") is None


class TestNames:
    def test_accumulates_nicks_across_several_353_lines(self):
        reg = IrcQueryRegistry()

        def send() -> None:
            reg.on_names_row("afternet", "#Chan", ["@alice", "+bob"])
            reg.on_names_row("afternet", "#chan", ["carol"])
            reg.on_names_end("afternet", "#CHAN")

        result = reg.names("afternet", "#chan", send, timeout=1.0)

        assert result is not None
        assert result.error is None
        assert result.nicks == ["@alice", "+bob", "carol"]

    def test_366_without_any_353_means_nothing_visible(self):
        """An unjoined +s channel answers with a bare 366; that is not an error."""
        reg = IrcQueryRegistry()

        def send() -> None:
            reg.on_names_end("afternet", "#secret")

        result = reg.names("afternet", "#secret", send, timeout=1.0)

        assert result is not None
        assert result.nicks == []
        assert result.error is None

    def test_403_no_such_channel_is_reported_as_error(self):
        reg = IrcQueryRegistry()

        def send() -> None:
            reg.on_error("afternet", "#nope", "No such channel")

        result = reg.names("afternet", "#nope", send, timeout=1.0)

        assert result is not None
        assert result.error == "No such channel"
        assert result.nicks == []

    def test_names_for_other_channels_do_not_leak_in(self):
        """A JOIN-triggered 353 for #other must not land in the #chan answer."""
        reg = IrcQueryRegistry()

        def send() -> None:
            reg.on_names_row("afternet", "#other", ["mallory"])
            reg.on_names_row("afternet", "#chan", ["alice"])
            reg.on_names_end("afternet", "#chan")

        result = reg.names("afternet", "#chan", send, timeout=1.0)

        assert result is not None
        assert result.nicks == ["alice"]

    def test_timeout_returns_none(self):
        reg = IrcQueryRegistry()
        assert reg.names("afternet", "#chan", lambda: None, timeout=0.01) is None

    def test_timeout_clears_the_pending_query_so_the_next_names_sends(self):
        """A server that never sends 366 must not wedge every later @names #chan."""
        reg = IrcQueryRegistry()
        sends: list[int] = []

        assert reg.names("afternet", "#chan", lambda: sends.append(1), timeout=0.01) is None
        assert reg.names("afternet", "#chan", lambda: sends.append(1), timeout=0.01) is None
        assert len(sends) == 2

    def test_concurrent_callers_share_one_in_flight_names(self):
        """Two askers during the same NAMES both get the answer to one request."""
        reg = IrcQueryRegistry()

        def answer() -> None:
            reg.on_names_row("afternet", "#chan", ["@alice", "bob"])
            reg.on_names_end("afternet", "#chan")

        sends, results = ask_twice(
            lambda send: reg.names("afternet", "#chan", send, timeout=2.0), answer
        )

        assert sends == 1
        assert [r.nicks if r else None for r in results] == [["@alice", "bob"]] * 2


class TestFormatChannels:
    def test_sorted_by_users_descending_and_capped(self):
        data = rows(("#small", 2, ""), ("#big", 40, "Big"), ("#mid", 9, "Mid"))
        out = ircquery.format_channels(data, limit=2)
        assert out == "2 of 3 channels: #big (40) Big | #mid (9) Mid"

    def test_glob_pattern_filters_case_insensitively(self):
        data = rows(("#Linux", 5, ""), ("#linuxhelp", 3, ""), ("#bsd", 8, ""))
        out = ircquery.format_channels(data, pattern="#linux*")
        assert out == "2 channels: #Linux (5) | #linuxhelp (3)"

    def test_exact_name_shows_that_channel_alone(self):
        data = rows(("#linux", 5, "Kernel talk"), ("#linuxhelp", 3, ""))
        out = ircquery.format_channels(data, pattern="#linux")
        assert out == "#linux (5) Kernel talk"

    def test_min_users_filter(self):
        data = rows(("#a", 1, ""), ("#b", 10, ""))
        out = ircquery.format_channels(data, min_users=5)
        assert out == "1 channel: #b (10)"

    def test_min_users_keeps_channels_with_exactly_that_many(self):
        """`@channels --min 5` means at least 5 users, per the command docs."""
        data = rows(("#a", 5, ""), ("#b", 4, ""))
        out = ircquery.format_channels(data, min_users=5)
        assert out == "1 channel: #a (5)"

    def test_no_match_says_so(self):
        assert ircquery.format_channels(rows(("#a", 1, "")), pattern="#zzz") == (
            "No channels match #zzz."
        )

    def test_topics_are_clipped(self):
        data = rows(
            ("#a", 1, "x" * 200),
        )
        out = ircquery.format_channels(data, topic_chars=20)
        assert out == "1 channel: #a (1) " + "x" * 19 + "…"

    def test_control_codes_in_topics_are_stripped(self):
        """Topics are third-party text bound for a channel line and the LLM."""
        data = rows(
            ("#a", 1, "\x02bold\x02 \x0304red\x03 ok"),
        )
        assert ircquery.format_channels(data) == "1 channel: #a (1) bold red ok"


class TestFormatNames:
    def test_lists_nicks_with_count(self):
        out = ircquery.format_names("#chan", ["@alice", "+bob", "carol"])
        assert out == "#chan (3): @alice +bob carol"

    def test_truncates_to_budget_with_remainder(self):
        nicks = [f"nick{i:02d}" for i in range(40)]
        out = ircquery.format_names("#chan", nicks, max_chars=60)
        assert len(out) <= 60
        assert out.startswith("#chan (40): nick00 nick01")
        assert out.endswith("more")

    def test_remainder_counts_exactly_the_nicks_left_out(self):
        """Shown nicks plus the `+N more` count must add up to the channel size."""
        nicks = [f"nick{i:02d}" for i in range(40)]
        out = ircquery.format_names("#chan", nicks, max_chars=60)
        body, _, suffix = out.removeprefix("#chan (40): ").rpartition(" +")
        shown = body.split()
        hidden = int(suffix.removesuffix(" more"))
        assert shown == nicks[: len(shown)]
        assert len(shown) + hidden == 40

    def test_empty_means_not_visible(self):
        assert ircquery.format_names("#chan", []) == (
            "#chan: no visible members (empty, secret, or private)."
        )


class TestWhois:
    def test_collects_the_standard_whois_numerics(self):
        reg = IrcQueryRegistry()

        def send() -> None:
            reg.on_whois_numeric("afternet", "311", ("Eck", "~eck", "host.example", "*", "Eck R."))
            reg.on_whois_numeric("afternet", "319", ("eck", "@#afternet +#help #linux"))
            reg.on_whois_numeric("afternet", "312", ("eck", "irc.afternet.org", "AfterNET"))
            reg.on_whois_numeric("afternet", "330", ("eck", "eckacct", "is logged in as"))
            reg.on_whois_numeric("afternet", "313", ("eck", "is an IRC operator"))
            reg.on_whois_numeric("afternet", "301", ("eck", "gone fishing"))
            reg.on_whois_numeric("afternet", "317", ("eck", "42", "1700000000", "seconds idle"))
            reg.on_whois_end("afternet", "ECK")

        result = reg.whois("afternet", "eck", send, timeout=1.0)

        assert result is not None
        assert result.error is None
        assert result.nick == "Eck"
        assert (result.user, result.host, result.realname) == ("~eck", "host.example", "Eck R.")
        assert result.channels == ["@#afternet", "+#help", "#linux"]
        assert (result.server, result.server_info) == ("irc.afternet.org", "AfterNET")
        assert result.account == "eckacct"
        assert result.oper is True
        assert result.away == "gone fishing"
        assert (result.idle_seconds, result.signon) == (42, 1700000000)

    def test_no_such_nick_is_an_error(self):
        reg = IrcQueryRegistry()

        def send() -> None:
            reg.on_error("afternet", "nobody", "No such nick/channel")

        result = reg.whois("afternet", "nobody", send, timeout=1.0)

        assert result is not None
        assert result.error == "No such nick/channel"

    def test_numerics_for_another_nick_do_not_leak_in(self):
        """The Network plugin's own @whois shares the wire; keep keys per nick."""
        reg = IrcQueryRegistry()

        def send() -> None:
            reg.on_whois_numeric("afternet", "311", ("other", "~o", "h", "*", "Other"))
            reg.on_whois_numeric("afternet", "311", ("eck", "~eck", "h", "*", "Eck"))
            reg.on_whois_end("afternet", "eck")

        result = reg.whois("afternet", "eck", send, timeout=1.0)

        assert result is not None
        assert result.realname == "Eck"

    def test_timeout_returns_none(self):
        reg = IrcQueryRegistry()
        assert reg.whois("afternet", "eck", lambda: None, timeout=0.01) is None

    def test_without_313_the_user_is_not_an_oper(self):
        """Only RPL_WHOISOPERATOR makes someone an IRC operator in the answer."""
        reg = IrcQueryRegistry()

        def send() -> None:
            reg.on_whois_numeric("afternet", "311", ("eck", "~eck", "h", "*", "Eck"))
            reg.on_whois_numeric("afternet", "312", ("eck", "irc.afternet.org", "AfterNET"))
            reg.on_whois_end("afternet", "eck")

        result = reg.whois("afternet", "eck", send, timeout=1.0)

        assert result is not None
        assert result.oper is False

    def test_timeout_clears_the_pending_query_so_the_next_whois_sends(self):
        """A server that never sends 318 must not wedge later lookups of that nick."""
        reg = IrcQueryRegistry()
        sends: list[int] = []

        assert reg.whois("afternet", "eck", lambda: sends.append(1), timeout=0.01) is None
        assert reg.whois("afternet", "eck", lambda: sends.append(1), timeout=0.01) is None
        assert len(sends) == 2

    def test_malformed_idle_is_ignored_not_fatal(self):
        reg = IrcQueryRegistry()

        def send() -> None:
            reg.on_whois_numeric("afternet", "317", ("eck", "soon", "seconds idle"))
            reg.on_whois_end("afternet", "eck")

        result = reg.whois("afternet", "eck", send, timeout=1.0)

        assert result is not None
        assert result.idle_seconds is None


class TestCollect:
    def test_returns_rows_fed_between_send_and_close(self):
        reg = IrcQueryRegistry()

        def send() -> None:
            reg.feed("who", "afternet", "#Chan", ("alice", "H"))
            reg.close("who", "afternet", "#chan")

        assert reg.collect("who", "afternet", "#chan", send, timeout=1.0) == (
            [("alice", "H")],
            None,
        )

    def test_timeout_returns_none_and_clears_the_pending_query(self):
        """Silence is an error, and it must not wedge later lookups of that target."""
        reg = IrcQueryRegistry()
        sends: list[int] = []

        result = reg.collect("who", "afternet", "#chan", lambda: sends.append(1), timeout=0.01)

        assert result is None
        assert reg.pending("who", "afternet", "#chan") is False
        assert (
            reg.collect("who", "afternet", "#chan", lambda: sends.append(1), timeout=0.01) is None
        )
        assert len(sends) == 2

    def test_a_timed_out_ctcp_wait_rejects_a_late_reply(self):
        """Nobody waits any more, so a stray CTCP VERSION reply is not accepted."""
        reg = IrcQueryRegistry()

        assert reg.collect("ctcp-VERSION", "afternet", "larry", lambda: None, timeout=0.01) is None

        assert reg.pending("ctcp-VERSION", "afternet", "larry") is False
        assert reg.feed("ctcp-VERSION", "afternet", "larry", "mIRC 7") is False

    def test_partial_returns_the_rows_that_arrived_before_the_timeout(self):
        """ADMIN with no 259: the LUSERS/VERSION rows that did arrive are kept."""
        reg = IrcQueryRegistry()

        def send() -> None:
            reg.feed("network", "afternet", "*", ("251", "There are 40 users"))

        result = reg.collect("network", "afternet", "*", send, timeout=0.01, partial=True)

        assert result == ([("251", "There are 40 users")], None)
        assert reg.pending("network", "afternet", "*") is False

    def test_partial_with_nothing_arrived_is_still_none(self):
        reg = IrcQueryRegistry()
        assert (
            reg.collect("network", "afternet", "*", lambda: None, timeout=0.01, partial=True)
            is None
        )

    def test_concurrent_callers_share_one_in_flight_query(self):
        """Two askers of the same (kind, target) both get the one reply."""
        reg = IrcQueryRegistry()

        def answer() -> None:
            reg.feed("who", "afternet", "#chan", ("alice", "H"))
            reg.close("who", "afternet", "#chan")

        sends, results = ask_twice(
            lambda send: reg.collect("who", "afternet", "#chan", send, timeout=2.0), answer
        )

        assert sends == 1
        assert results == [([("alice", "H")], None)] * 2
