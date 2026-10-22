"""TypingHolds: one +typing indicator per (network, target), refcounted."""

from __future__ import annotations

import logging
import threading
import time
from types import SimpleNamespace

import pytest
from llm.typing_holds import TypingHolds


class _Recorder:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str, str]] = []
        self.lock = threading.Lock()

    def __call__(self, irc, target: str, state: str) -> None:
        with self.lock:
            self.sent.append((irc.network, target, state))

    def states(self, network: str, target: str) -> list[str]:
        with self.lock:
            return [s for n, t, s in self.sent if n == network and t == target]


@pytest.fixture
def rec() -> _Recorder:
    return _Recorder()


@pytest.fixture
def holds(rec: _Recorder) -> TypingHolds:
    h = TypingHolds(rec, logging.getLogger("test"), interval=0.05)
    yield h
    h.stop()


def _irc(network: str = "afternet") -> SimpleNamespace:
    return SimpleNamespace(network=network)


class TestRefcount:
    def test_second_hold_on_same_target_sends_nothing(self, holds, rec) -> None:
        r1 = holds.hold(_irc(), "#chan")
        r2 = holds.hold(_irc(), "#chan")
        assert rec.states("afternet", "#chan") == ["active"]
        r1()
        assert rec.states("afternet", "#chan") == ["active"], "done must wait for the last holder"
        r2()
        assert rec.states("afternet", "#chan") == ["active", "done"]

    def test_release_is_idempotent(self, holds, rec) -> None:
        r1 = holds.hold(_irc(), "#chan")
        r2 = holds.hold(_irc(), "#chan")
        r1()
        r1()
        r1()
        assert holds.holds("afternet", "#chan")
        r2()
        assert not holds.holds("afternet", "#chan")
        assert rec.states("afternet", "#chan").count("done") == 1

    def test_same_channel_on_two_networks_is_two_indicators(self, holds, rec) -> None:
        ra = holds.hold(_irc("afternet"), "#chan")
        rb = holds.hold(_irc("other"), "#chan")
        assert holds.active_targets() == {("afternet", "#chan"), ("other", "#chan")}
        ra()
        assert rec.states("afternet", "#chan") == ["active", "done"]
        assert rec.states("other", "#chan") == ["active"]
        rb()


class TestKeepalive:
    def test_active_is_resent_while_held(self, holds, rec) -> None:
        release = holds.hold(_irc(), "#chan")
        time.sleep(0.3)
        release()
        assert rec.states("afternet", "#chan").count("active") >= 2
        assert rec.states("afternet", "#chan")[-1] == "done"

    def test_nothing_is_sent_after_release(self, holds, rec) -> None:
        release = holds.hold(_irc(), "#chan")
        release()
        n = len(rec.states("afternet", "#chan"))
        time.sleep(0.15)
        assert len(rec.states("afternet", "#chan")) == n

    def test_a_stalled_keepalive_never_lands_after_the_last_done(self, rec) -> None:
        """The TLC counterexample (docs/formal/TypingHolds.tla): the keepalive
        picks the target while it is held, the last holder releases and sends
        ``done``, then the keepalive's ``active`` lands and the client shows
        typing after the reply."""
        gate, stalled = threading.Event(), threading.Event()

        def send(irc, target: str, state: str) -> None:
            if threading.current_thread().name == "typing-keepalive" and not gate.is_set():
                stalled.set()
                gate.wait(5)
            rec(irc, target, state)

        holds = TypingHolds(send, logging.getLogger("test"), interval=0.05)
        try:
            release = holds.hold(_irc(), "#chan")
            assert stalled.wait(5), "keepalive never fired"
            releaser = threading.Thread(target=release)
            releaser.start()
            time.sleep(0.1)  # let the release race ahead of the stalled send
            gate.set()
            releaser.join(5)
            time.sleep(0.2)
            assert rec.states("afternet", "#chan")[-1] == "done"
        finally:
            holds.stop()

    def test_send_failure_does_not_kill_the_loop(self, rec) -> None:
        calls = {"n": 0}

        def flaky(irc, target, state):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("boom")
            rec(irc, target, state)

        h = TypingHolds(flaky, logging.getLogger("test"), interval=0.02)
        try:
            release = h.hold(_irc(), "#chan")
            time.sleep(0.15)
            release()
        finally:
            h.stop()
        assert rec.states("afternet", "#chan").count("active") >= 2


class TestSingleThread:
    def test_every_hold_shares_one_keepalive_thread(self, holds) -> None:
        """GIVEN holds taken one after another and overlapping THEN one thread serves them all.

        The module promises a single keepalive thread. A thread per hold would
        leak one looping thread per reply until shutdown, and on a small box
        end in ``can't start new thread``.
        """
        before = {t for t in threading.enumerate() if t.name == "typing-keepalive"}

        holds.hold(_irc(), "#chan")()
        holds.hold(_irc(), "#chan")()
        overlapping = [holds.hold(_irc(), "#chan"), holds.hold(_irc(), "#other")]
        for release in overlapping:
            release()

        ours = [
            t
            for t in threading.enumerate()
            if t.name == "typing-keepalive" and t.is_alive() and t not in before
        ]
        assert len(ours) == 1


class TestThreadStartFailure:
    def test_a_thread_that_will_not_start_does_not_strand_the_refcount(self, rec) -> None:
        """GIVEN the keepalive thread fails to start WHEN acquiring THEN the count unwinds.

        A refcount left at one with no thread means the last release never
        fires, and that target swallows every ``done`` for the life of the
        process.

        A long interval, so the only ``active`` here is the one the acquire
        sends: a keepalive landing mid-test would make the count meaningless.
        """
        h = TypingHolds(rec, logging.getLogger("test"), interval=30)
        real_start = threading.Thread.start
        calls = {"n": 0}

        def flaky_start(self) -> None:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("can't start new thread")
            real_start(self)

        try:
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(threading.Thread, "start", flaky_start)
                with pytest.raises(RuntimeError, match="can't start new thread"):
                    h.hold(_irc(), "#chan")

            assert not h.holds("afternet", "#chan")
            assert rec.states("afternet", "#chan") == []

            release = h.hold(_irc(), "#chan")

            assert h.holds("afternet", "#chan")
            assert rec.states("afternet", "#chan") == ["active"]

            release()
            assert rec.states("afternet", "#chan") == ["active", "done"]
        finally:
            h.stop()


class TestStop:
    def test_stop_ends_the_thread_and_sends_nothing_more(self, rec) -> None:
        h = TypingHolds(rec, logging.getLogger("test"), interval=0.02)
        h.hold(_irc(), "#chan")
        h.stop()
        n = len(rec.sent)
        time.sleep(0.1)
        assert len(rec.sent) == n
        assert not any(t.name == "typing-keepalive" and t.is_alive() for t in threading.enumerate())

    def test_a_hold_taken_after_stop_does_not_restart_the_thread(self, rec) -> None:
        """A worker finishing after die() must not resurrect the keepalive."""
        h = TypingHolds(rec, logging.getLogger("test"), interval=0.02)
        h.stop()

        h.hold(_irc(), "#chan")

        assert not any(t.name == "typing-keepalive" and t.is_alive() for t in threading.enumerate())
