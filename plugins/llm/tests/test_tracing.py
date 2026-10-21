"""Tests for request tracing module."""

from __future__ import annotations

import logging
import re
import threading

import pytest
import supybot.registry as registry
from llm.tracing import TraceFilter, generate_request_id, request_id


class TestGenerateRequestId:
    """Tests for generate_request_id()."""

    def test_returns_8_hex_chars(self) -> None:
        """GIVEN nothing WHEN generate_request_id THEN returns 8 hex characters."""
        rid = generate_request_id()
        assert len(rid) == 8
        assert re.fullmatch(r"[0-9a-f]{8}", rid)

    def test_returns_unique_values(self) -> None:
        """GIVEN multiple calls WHEN generate_request_id THEN each value is unique."""
        ids = {generate_request_id() for _ in range(100)}
        assert len(ids) == 100


class TestTraceFilter:
    """Tests for TraceFilter logging filter."""

    def test_prepends_id_when_set(self) -> None:
        """GIVEN request_id is set WHEN filter runs THEN message is prefixed with [id]."""
        trace_filter = TraceFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="hello world",
            args=(),
            exc_info=None,
        )

        token = request_id.set("abc12345")
        try:
            result = trace_filter.filter(record)
            assert result is True
            assert record.msg == "[abc12345] hello world"
        finally:
            request_id.reset(token)

    def test_leaves_message_unchanged_when_empty(self) -> None:
        """GIVEN request_id is default (empty) WHEN filter runs THEN message unchanged."""
        trace_filter = TraceFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="hello world",
            args=(),
            exc_info=None,
        )

        result = trace_filter.filter(record)
        assert result is True
        assert record.msg == "hello world"

    def test_always_returns_true(self) -> None:
        """GIVEN any state WHEN filter runs THEN returns True (never suppresses)."""
        trace_filter = TraceFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test",
            args=(),
            exc_info=None,
        )
        assert trace_filter.filter(record) is True


class TestThreadIsolation:
    """Tests for ContextVar thread isolation."""

    def test_threads_see_own_trace_ids(self) -> None:
        """GIVEN two threads with different trace IDs WHEN each reads request_id THEN isolated."""
        results: dict[str, str] = {}
        barrier = threading.Barrier(2)

        def worker(name: str, rid: str) -> None:
            token = request_id.set(rid)
            barrier.wait()  # Ensure both threads are running concurrently
            results[name] = request_id.get()
            request_id.reset(token)

        t1 = threading.Thread(target=worker, args=("t1", "aaaa1111"))
        t2 = threading.Thread(target=worker, args=("t2", "bbbb2222"))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert results["t1"] == "aaaa1111"
        assert results["t2"] == "bbbb2222"

    def test_default_is_empty_in_new_thread(self) -> None:
        """GIVEN request_id set in main thread WHEN new thread reads THEN gets default."""
        token = request_id.set("main1234")
        result: list[str] = []

        def worker() -> None:
            result.append(request_id.get())

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        request_id.reset(token)

        assert result[0] == ""


class TestValidatedLogLevel:
    """Tests for ValidatedLogLevel registry type."""

    def test_accepts_warning(self) -> None:
        """GIVEN 'WARNING' WHEN set THEN accepted."""
        from llm.config import ValidatedLogLevel

        v = ValidatedLogLevel("WARNING", "test")
        v.setValue("WARNING")
        assert v() == "WARNING"

    def test_accepts_debug(self) -> None:
        """GIVEN 'DEBUG' WHEN set THEN accepted."""
        from llm.config import ValidatedLogLevel

        v = ValidatedLogLevel("WARNING", "test")
        v.setValue("DEBUG")
        assert v() == "DEBUG"

    def test_accepts_lowercase(self) -> None:
        """GIVEN 'debug' WHEN set THEN normalized to 'DEBUG'."""
        from llm.config import ValidatedLogLevel

        v = ValidatedLogLevel("WARNING", "test")
        v.setValue("debug")
        assert v() == "DEBUG"

    @pytest.mark.parametrize("value", ["VERBOSE", "3", "TRACE", ""])
    def test_rejects_invalid(self, value: str) -> None:
        """GIVEN invalid level WHEN set THEN raises InvalidRegistryValue."""
        from llm.config import ValidatedLogLevel

        v = ValidatedLogLevel("WARNING", "test")
        with pytest.raises(registry.InvalidRegistryValue):
            v.setValue(value)
