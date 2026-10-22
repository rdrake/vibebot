"""Global concurrency cap for all LLM I/O in the plugin.

Both command-path and background callers funnel through here so a
single ``maxConcurrentLLMCalls`` knob bounds outbound provider load
regardless of which thread originated the call.

Two surfaces:

* ``permit()`` — context manager. Use from a thread that already
  exists (Limnoria CommandThread). Acquires/releases one slot from
  the bounded semaphore.
* ``submit(label, fn, *args, **kw) -> Future`` — runs ``fn`` on a
  pool worker after acquiring a slot. Use from the main IRC thread
  inside a ``schedule.addEvent`` shim so the main thread returns
  immediately.

Workers must NEVER call :meth:`submit` recursively — saturating the
pool would deadlock. ``RecursiveSubmitError`` is raised at runtime as
a guardrail. Note: the guard fires at ``submit()`` call time (outside
the worker function body), not inside the worker wrapper. Keep it
that way — putting it inside the wrapper would not catch the case
where ``submit`` is called from worker context but never executes.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Generator
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from logging import Logger
from typing import Any

from .tracing import request_id

_WORKER_THREAD_PREFIX = "llm-worker"


class RecursiveSubmitError(RuntimeError):
    """Raised when LLMExecutor.submit is called from a pool worker."""


class LLMExecutor:
    def __init__(self, max_concurrency: int, log: Logger) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self._max = max_concurrency
        self._log = log
        self._semaphore = threading.BoundedSemaphore(max_concurrency)
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrency,
            thread_name_prefix=_WORKER_THREAD_PREFIX,
        )
        self._running = 0
        self._queued = 0
        self._counter_lock = threading.Lock()
        self._closing = False
        # Track in-flight futures so drain() can wait without re-shutting down.
        self._inflight: set[Future[Any]] = set()
        self._inflight_lock = threading.Lock()
        # Worker-context flag — robust against thread renaming/wrapping
        # (a thread-name-prefix check silently misfires if anything ever
        # wraps the pool or renames threads). Set inside `_run` below.
        self._worker_ctx = threading.local()

    @property
    def max_concurrency(self) -> int:
        return self._max

    @property
    def closing(self) -> bool:
        return self._closing

    def running(self) -> int:
        with self._counter_lock:
            return self._running

    def queued(self) -> int:
        with self._counter_lock:
            return self._queued

    @contextmanager
    def permit(self) -> Generator[None]:
        self._semaphore.acquire()
        with self._counter_lock:
            self._running += 1
        try:
            yield
        finally:
            with self._counter_lock:
                self._running -= 1
            self._semaphore.release()

    def submit(
        self,
        label: str,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Future[Any]:
        if getattr(self._worker_ctx, "in_worker", False):
            raise RecursiveSubmitError(
                f"submit('{label}') called from worker thread; nested LLM "
                "calls must run inline within the existing permit"
            )
        captured_rid = request_id.get()
        with self._counter_lock:
            self._queued += 1
        submit_t = time.monotonic()
        self._log.info(
            "llm_executor submit label=%s running=%s queued=%s max=%s",
            label,
            self._running,
            self._queued,
            self._max,
        )

        def _run() -> Any:
            self._worker_ctx.in_worker = True
            self._semaphore.acquire()
            with self._counter_lock:
                self._queued -= 1
                self._running += 1
            token = request_id.set(captured_rid) if captured_rid else None
            start = time.monotonic()
            try:
                return fn(*args, **kwargs)
            except Exception:
                self._log.exception("llm_executor task raised: label=%s", label)
                raise
            finally:
                elapsed_ms = int((time.monotonic() - start) * 1000)
                self._log.info(
                    "llm_executor done label=%s elapsed_ms=%s queued_ms=%s",
                    label,
                    elapsed_ms,
                    int((start - submit_t) * 1000),
                )
                if token is not None:
                    request_id.reset(token)
                with self._counter_lock:
                    self._running -= 1
                self._semaphore.release()
                # Clear the in_worker flag — pool threads are reused, so
                # leaving it set would falsely trip the guard for the
                # next task on the same thread.
                self._worker_ctx.in_worker = False

        try:
            fut = self._executor.submit(_run)
        except BaseException:
            # Pool dispatch failed (e.g. RuntimeError from a submit that raced
            # shutdown via the non-atomic `closing` gate). The matching
            # ``_queued -= 1`` lives inside ``_run``, which never executes on
            # this path — roll the gauge back so it doesn't leak for the
            # executor's lifetime. BaseException so KeyboardInterrupt/SystemExit
            # also restore the counter before propagating.
            with self._counter_lock:
                self._queued -= 1
            raise
        with self._inflight_lock:
            self._inflight.add(fut)
        fut.add_done_callback(self._discard_inflight)
        return fut

    def _discard_inflight(self, fut: Future[Any]) -> None:
        with self._inflight_lock:
            self._inflight.discard(fut)

    def drain(self, timeout: float) -> bool:
        """Wait up to ``timeout`` seconds for in-flight tasks to finish.

        Returns True if all in-flight tasks completed within the budget,
        False otherwise. Does NOT shut down the executor — callers do
        that via :meth:`shutdown` (typically before drain so queued
        futures are cancelled and only the actually-running tasks are
        awaited).
        """
        with self._inflight_lock:
            futs = set(self._inflight)
        if not futs:
            return True
        done, not_done = wait(futs, timeout=timeout)
        return not not_done

    def shutdown(self) -> None:
        """Mark closing and cancel queued futures. Idempotent."""
        if self._closing:
            return
        self._closing = True
        self._executor.shutdown(cancel_futures=True, wait=False)
