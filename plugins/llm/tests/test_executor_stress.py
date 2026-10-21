"""Stress and regression tests for the LLM executor migration."""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import wait

import pytest
from llm.executor import LLMExecutor


def test_cap_honored_under_burst() -> None:
    """Cap is fully utilized AND not exceeded.

    A `<=` assertion would pass even if the cap were 1 (or 0 — never
    observed running). Use a barrier so all `max_concurrency` workers
    reach `running()` measurement together, then assert exact equality.
    """
    ex = LLMExecutor(max_concurrency=4, log=logging.getLogger("test"))
    try:
        max_seen = [0]
        seen_lock = threading.Lock()  # SHARED, not per-task — bug from v1 plan.
        all_started = threading.Barrier(ex.max_concurrency)
        release = threading.Event()

        def task() -> None:
            try:
                # Wait for all `max_concurrency` workers to be running
                # before sampling, so `running()` is measured at the
                # saturated state.
                all_started.wait(timeout=5)
            except threading.BrokenBarrierError:
                return
            with seen_lock:
                max_seen[0] = max(max_seen[0], ex.running())
            release.wait(timeout=5)

        # Submit max_concurrency * 3 so queue depth is also exercised.
        futs = [ex.submit(f"t{i}", task) for i in range(ex.max_concurrency * 3)]
        # Wait for the barrier to release all initial workers, then
        # give the seen-lock window a moment to capture.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and max_seen[0] < ex.max_concurrency:
            time.sleep(0.02)
        assert max_seen[0] == ex.max_concurrency, (
            f"expected {ex.max_concurrency} concurrent, saw {max_seen[0]}"
        )
        release.set()
        done, not_done = wait(futs, timeout=5)
        assert not not_done, "all tasks must finish"
        assert all(f.exception() is None for f in done)
    finally:
        ex.shutdown()


def test_safe_queue_serializes_under_load(plugin_env, mocker) -> None:
    plugin, _irc, _msg = plugin_env
    target = mocker.MagicMock()

    def call() -> None:
        plugin._safe_queue(target, mocker.MagicMock())

    threads = [threading.Thread(target=call) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert target.queueMsg.call_count == 50


def test_reload_drops_post_completion_writes(plugin_env, mocker) -> None:
    """Slow worker is in-flight; die() called; worker tail-end writes short-circuit."""
    plugin, irc, _msg = plugin_env
    started = threading.Event()
    release = threading.Event()
    db_writes: list[object] = []

    plugin.db.log_usage = mocker.MagicMock(side_effect=lambda *a, **kw: db_writes.append(a))

    def slow_task() -> None:
        started.set()
        release.wait(timeout=5)
        # Simulated tail-end behaviour gated by closing.
        if plugin._llm_executor.closing:
            return
        plugin._safe_queue(irc, mocker.sentinel.msg)
        plugin.db.log_usage("alice", "#chan", "ask", "model", 1, 1, 0.0)

    fut = plugin._llm_executor.submit("slow", slow_task)
    assert started.wait(timeout=2)

    plugin.die()
    release.set()
    # Deterministically wait for the worker's tail-end to run instead of
    # a fixed sleep. die() already drained (timed out while the worker
    # was blocked on `release`); now that release is set the worker
    # returns normally under the closing gate, so result() returns once
    # the tail-end has executed.
    fut.result(timeout=2)

    irc.queueMsg.assert_not_called()
    assert not db_writes, "post-die() writes must be suppressed by closing gate"


@pytest.mark.parametrize("concurrency", [1, 4, 8])
def test_drain_after_shutdown_with_pending(concurrency: int) -> None:
    """shutdown(cancel_futures=True) cancels queued tasks; drain awaits running."""
    ex = LLMExecutor(max_concurrency=concurrency, log=logging.getLogger("test"))
    release = threading.Event()
    # Saturate the pool, then queue extras.
    running_futs = [ex.submit(f"r{i}", release.wait, 5) for i in range(concurrency)]
    queued_futs = [ex.submit(f"q{i}", release.wait, 5) for i in range(concurrency * 2)]
    time.sleep(0.05)
    ex.shutdown()
    # Queued futures should be cancelled, running ones still running.
    assert any(f.cancelled() for f in queued_futs)
    threading.Timer(0.1, release.set).start()
    assert ex.drain(timeout=3.0) is True
    for f in running_futs:
        assert f.done()
