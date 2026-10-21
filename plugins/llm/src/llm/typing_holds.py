"""One IRCv3 +typing indicator per (network, target), shared by every holder.

Every path that wants the bot to look busy on a target — a command's planner
turn, a chat reply — takes a *hold*. Holds are refcounted per ``(network, target)``: the first sends
``+typing=active``, the last release sends ``+typing=done``, and everything in
between is silent. A single daemon thread re-sends ``active`` for every held
target every ``interval`` seconds, because clients drop the state after about
six seconds without a refresh.

The network is part of the key because ``#chan`` on two networks is two rooms.

Sends run outside ``_lock`` (a send can block on the IRC send lock behind a
slow driver, and every ``hold()`` would stall with it) but inside
``_send_lock``, and each one reads the key's state at send time rather than
carrying the state it was triggered with. Otherwise a keepalive ``active``
snapshotted before the last release lands after that release's ``done``, and
the client shows typing after the reply. See docs/formal/TypingHolds*.tla.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

Key = tuple[str, str]


class TypingHolds:
    def __init__(
        self,
        send: Callable[[Any, str, str], None],
        log: Any,
        interval: float = 4.0,
    ) -> None:
        self._send = send
        self._log = log
        self._interval = interval
        self._lock = threading.Lock()
        # Serializes every send; taken before _lock, never inside it.
        self._send_lock = threading.Lock()
        self._counts: dict[Key, int] = {}
        self._ircs: dict[Key, Any] = {}
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- public -----------------------------------------------------------

    def hold(self, irc: Any, target: str) -> Callable[[], None]:
        """Take one hold on ``target``; returns an idempotent release."""
        key = (getattr(irc, "network", "") or "", target)
        self._acquire(key, irc)
        released = False
        lock = threading.Lock()

        def release() -> None:
            nonlocal released
            with lock:
                if released:
                    return
                released = True
            self._release(key)

        return release

    def holds(self, network: str, target: str) -> bool:
        with self._lock:
            return (network, target) in self._counts

    def active_targets(self) -> set[Key]:
        with self._lock:
            return set(self._counts)

    def stop(self) -> None:
        """Stop the keepalive thread. Sends nothing: callers stop at shutdown,
        when the send path is already closed, and clients expire the state."""
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
            if thread.is_alive():
                self._log.warning("typing: keepalive thread still alive after stop()")

    # -- internals --------------------------------------------------------

    def _acquire(self, key: Key, irc: Any) -> None:
        with self._lock:
            self._ircs[key] = irc
            self._counts[key] = self._counts.get(key, 0) + 1
            first = self._counts[key] == 1
            try:
                self._ensure_thread()
            except Exception:
                # A thread that would not start (``can't start new thread``)
                # must not leave the refcount raised: a count stuck above
                # zero would mean the last release never fires and that
                # target swallows every ``done`` from here on.
                remaining = self._counts.get(key, 0) - 1
                if remaining <= 0:
                    self._counts.pop(key, None)
                    self._ircs.pop(key, None)
                else:
                    self._counts[key] = remaining
                raise
        if first:
            self._send_current(key, irc)
            self._wake.set()

    def _release(self, key: Key) -> None:
        with self._lock:
            count = self._counts.get(key, 0)
            if count <= 1:
                self._counts.pop(key, None)
                irc = self._ircs.pop(key, None)
                last = count == 1
            else:
                self._counts[key] = count - 1
                irc = None
                last = False
        if last and irc is not None:
            self._send_current(key, irc)

    def _send_current(self, key: Key, fallback_irc: Any) -> None:
        """Send whatever ``key`` is now: ``active`` if held, else ``done``.

        A transition that lost a race sends the winner's state, which is a
        harmless duplicate of the winner's own send; what it can never do is
        land a stale state after a newer one.
        """
        with self._send_lock:
            with self._lock:
                held = key in self._counts
                irc = self._ircs.get(key, fallback_irc)
            self._safe_send(irc, key[1], "active" if held else "done")

    def _keepalive(self, key: Key) -> None:
        """Re-send ``active`` for ``key`` only if it is still held."""
        with self._send_lock:
            with self._lock:
                irc = self._ircs.get(key)
            if irc is not None:
                self._safe_send(irc, key[1], "active")

    def _safe_send(self, irc: Any, target: str, state: str) -> None:
        try:
            self._send(irc, target, state)
        except Exception:
            self._log.exception("typing: %s send failed for %s", state, target)

    def _ensure_thread(self) -> None:
        """Start the keepalive thread on first use. Caller holds ``_lock``."""
        if self._thread is not None and self._thread.is_alive():
            return
        if self._stop.is_set():
            return
        self._thread = threading.Thread(target=self._loop, name="typing-keepalive", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            if not self._wake.wait(timeout=1.0):
                continue
            while not self._stop.is_set():
                with self._lock:
                    if not self._ircs:
                        self._wake.clear()
                        break
                if self._stop.wait(timeout=self._interval):
                    return
                with self._lock:
                    keys = list(self._ircs)
                for key in keys:
                    self._keepalive(key)
