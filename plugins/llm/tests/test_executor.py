"""Tests for the LLMExecutor — global cap on LLM I/O concurrency."""

from __future__ import annotations

import logging
import threading
import time

import pytest
from llm.executor import LLMExecutor, RecursiveSubmitError
from llm.tracing import request_id


class TestLLMExecutorBasics:
    def test_submit_runs_function(self) -> None:
        ex = LLMExecutor(max_concurrency=2, log=logging.getLogger("test"))
        try:
            fut = ex.submit("test", lambda: 42)
            assert fut.result(timeout=2) == 42
        finally:
            ex.shutdown()

    def test_permit_acquires_and_releases(self) -> None:
        ex = LLMExecutor(max_concurrency=1, log=logging.getLogger("test"))
        try:
            entered = threading.Event()
            release = threading.Event()

            def hold() -> None:
                with ex.permit():
                    entered.set()
                    release.wait(timeout=2)

            t = threading.Thread(target=hold)
            t.start()
            assert entered.wait(timeout=1)
            assert ex.running() == 1

            second_acquired = threading.Event()

            def try_acquire() -> None:
                with ex.permit():
                    second_acquired.set()

            t2 = threading.Thread(target=try_acquire)
            t2.start()
            assert not second_acquired.wait(timeout=0.2)

            release.set()
            t.join(timeout=2)
            assert second_acquired.wait(timeout=1)
            t2.join(timeout=2)
        finally:
            ex.shutdown()

    def test_submit_swallows_and_logs_exceptions(self, caplog: pytest.LogCaptureFixture) -> None:
        ex = LLMExecutor(max_concurrency=2, log=logging.getLogger("test"))
        try:
            with caplog.at_level(logging.ERROR):
                fut = ex.submit("bad", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
                with pytest.raises(RuntimeError):
                    fut.result(timeout=2)
            assert any("bad" in r.getMessage() for r in caplog.records)
        finally:
            ex.shutdown()

    def test_submit_after_shutdown_does_not_leak_queued_counter(self) -> None:
        """A submit that races shutdown must not leave ``_queued`` over-counted.

        ``_queued`` is incremented at submit() call time but decremented inside
        the worker body, which never runs if the pool dispatch itself raises
        (RuntimeError after shutdown). The increment must be rolled back.
        """
        ex = LLMExecutor(max_concurrency=1, log=logging.getLogger("test"))
        ex.shutdown()
        assert ex.queued() == 0
        with pytest.raises(RuntimeError):
            ex.submit("after-shutdown", lambda: 1)
        assert ex.queued() == 0

    def test_running_and_queued_counters(self) -> None:
        ex = LLMExecutor(max_concurrency=1, log=logging.getLogger("test"))
        try:
            release = threading.Event()
            futs = [ex.submit(f"t{i}", release.wait, 5) for i in range(3)]
            for _ in range(50):
                if ex.running() == 1:
                    break
                time.sleep(0.02)
            assert ex.running() == 1
            assert ex.queued() == 2
            release.set()
            for f in futs:
                f.result(timeout=2)
        finally:
            ex.shutdown()

    def test_request_id_propagation_with_reset(self) -> None:
        ex = LLMExecutor(max_concurrency=1, log=logging.getLogger("test"))
        try:
            seen: list[str | None] = []

            def capture() -> None:
                seen.append(request_id.get())

            tok = request_id.set("trace-A")
            try:
                ex.submit("a", capture).result(timeout=2)
            finally:
                request_id.reset(tok)

            ex.submit("b", capture).result(timeout=2)
            assert seen[0] == "trace-A"
            assert seen[1] != "trace-A"
        finally:
            ex.shutdown()

    def test_recursive_submit_from_worker_raises(self) -> None:
        ex = LLMExecutor(max_concurrency=2, log=logging.getLogger("test"))
        try:
            captured: list[Exception] = []

            def inner() -> None:
                try:
                    ex.submit("inner", lambda: None)
                except RecursiveSubmitError as e:
                    captured.append(e)

            ex.submit("outer", inner).result(timeout=2)
            assert captured
        finally:
            ex.shutdown()

    def test_drain_waits_for_in_flight(self) -> None:
        """drain(timeout) waits for in-flight tasks; finished within budget => True."""
        ex = LLMExecutor(max_concurrency=2, log=logging.getLogger("test"))
        release = threading.Event()
        ex.submit("slow", release.wait, 5)
        time.sleep(0.05)
        # Schedule release after a short delay; drain should return True.
        threading.Timer(0.2, release.set).start()
        assert ex.drain(timeout=2.0) is True
        ex.shutdown()

    def test_drain_returns_false_on_timeout(self) -> None:
        ex = LLMExecutor(max_concurrency=1, log=logging.getLogger("test"))
        release = threading.Event()
        ex.submit("very-slow", release.wait, 30)
        time.sleep(0.05)
        assert ex.drain(timeout=0.2) is False
        release.set()
        ex.shutdown()

    def test_shutdown_returns_without_waiting_for_running_task(self) -> None:
        """shutdown() must not block on a running call.

        die() calls shutdown() then drain(timeout=2.0); the bounded wait
        belongs to drain. If shutdown waited for workers, one hung provider
        call would hang unload/reload/quit past that budget.
        """
        ex = LLMExecutor(max_concurrency=1, log=logging.getLogger("test"))
        started = threading.Event()
        release = threading.Event()

        def hang() -> None:
            started.set()
            release.wait(10)

        fut = ex.submit("hung", hang)
        try:
            assert started.wait(timeout=2)
            t0 = time.monotonic()
            ex.shutdown()
            elapsed = time.monotonic() - t0
            assert elapsed < 0.5
            assert fut.running()
            assert ex.drain(timeout=0.1) is False
        finally:
            release.set()
        assert ex.drain(timeout=2.0) is True
