"""Request tracing for LLM plugin."""

from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar

request_id: ContextVar[str] = ContextVar("request_id", default="")


def generate_request_id() -> str:
    """Generate a short unique request ID (8 hex chars)."""
    return uuid.uuid4().hex[:8]


class TraceFilter(logging.Filter):
    """Logging filter that prepends the current request ID to log messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        rid = request_id.get()
        if rid:
            record.msg = f"[{rid}] {record.msg}"
        return True
