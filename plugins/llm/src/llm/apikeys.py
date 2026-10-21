"""Provider-scoped API key resolution.

A key is a property of which provider is being paid, not of which channel or
which role asked. This module maps a model identifier to its provider and then
to a single environment variable, so the key is always anchored to the model
actually being called.

Pure and dependency-light on purpose: no plugin, no service, no Limnoria
registry. That keeps it unit-testable and keeps key resolution out of
``service.py``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping

# The providers this deployment pays for: ``provider/name`` model strings
# with any other prefix have no key and no client (see provider.py).
PROVIDER_ENV_VARS: dict[str, str] = {
    "xai": "XAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def provider_of(model: str) -> str:
    """Return the provider prefix of ``model``, or "" if it names none we call.

    Never raises: callers sit inside failure handlers, where a new exception
    type would surface as a crash rather than a configuration error.
    """
    prefix, sep, name = (model or "").strip().partition("/")
    if not sep or not name:
        return ""
    prefix = prefix.lower()
    return prefix if prefix in PROVIDER_ENV_VARS else ""


def env_var_for(model: str) -> str | None:
    """Environment variable name ``model`` needs, or None if unmanaged.

    Returns the *name*, never the value, so callers can tell an operator which
    variable to set.
    """
    return PROVIDER_ENV_VARS.get(provider_of(model))


def is_managed(model: str) -> bool:
    """True if ``model`` names a provider this deployment has a key for."""
    return provider_of(model) in PROVIDER_ENV_VARS


def api_key_for(model: str) -> str | None:
    """Configured API key for ``model``'s provider, or None.

    Read from the environment on every call rather than cached, so the value
    redaction scrubs can never diverge from the value actually sent.
    """
    name = env_var_for(model)
    if not name:
        return None
    return os.environ.get(name, "").strip() or None


# Credentials do not all end in _API_KEY. A provider whose secret is named
# differently would otherwise sit outside redaction entirely.
SECRET_SUFFIXES: tuple[str, ...] = ("_API_KEY", "_TOKEN", "_SECRET", "_CREDENTIALS")

# Real keys are far longer (Gemini 39, xAI ~84). The floor stops a short junk
# value such as FOO_API_KEY=disabled turning redaction into find-and-replace on
# a common word across every log line.
MIN_REDACTABLE_LEN = 16

REDACTED = "[REDACTED]"


def _secret_items() -> list[tuple[str, str]]:
    """(name, stripped value) for every environment secret worth redacting.

    Enumerates *names* only (``list(os.environ)``, a genuinely atomic key
    snapshot) and then reads each one with ``.get()``, which is
    ``try: return self[key] except KeyError: return default`` — so a
    concurrent ``setenv``/``delenv`` between the enumeration and the read
    just yields the default rather than raising. That makes this race-free by
    construction: no retry loop, no fallback, no window where redaction goes
    quiet because the environment happened to change underneath it.
    """
    items = []
    for name in list(os.environ):
        if not name.upper().endswith(SECRET_SUFFIXES):
            continue
        value = os.environ.get(name, "").strip()
        if len(value) >= MIN_REDACTABLE_LEN:
            items.append((name, value))
    return items


def known_secret_values() -> set[str]:
    """Every environment value that must never appear in output."""
    return {value for _name, value in _secret_items()}


def secret_var_names() -> list[str]:
    """Sorted names of the variables redaction covers. Names only, never values."""
    return sorted(name for name, _value in _secret_items())


def scrub(text: str | None) -> str:
    """Replace every known secret value in ``text`` with ``[REDACTED]``."""
    return _scrub_with(text, known_secret_values())


def _scrub_with(text: str | None, secrets: set[str]) -> str:
    """Same replacement as :func:`scrub`, against an already-collected secret set.

    :class:`SecretFilter` scrubs up to four fields per record (message, args,
    traceback, stack info); calling :func:`scrub` for each would recompute
    ``known_secret_values()`` — an environment scan — every time. Filter
    collects the set once per record and threads it through this instead.
    """
    if not text:
        return ""
    result = str(text)
    for secret in secrets:
        result = result.replace(secret, REDACTED)
    return result


class SecretFilter(logging.Filter):
    """Strip API keys from log records before they are formatted.

    Covers message, arguments, traceback and stack info, because supybot's
    ``Logger.exception`` uses all of them: it writes the raw traceback and calls
    ``collect_extra_debug_data()``, a repr of every frame local and every
    attribute of ``self``.

    Arguments are scrubbed by their ``str()`` value rather than by type: an
    exception object carries the provider's error body, which routinely echoes
    the submitted key.

    Never drops a record, and never breaks formatting — redaction must not cost
    observability. Value replacement is defence in depth, not a boundary: it
    cannot catch an encoded or truncated credential.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        secrets = known_secret_values()
        if not secrets:
            return True
        record.msg = _scrub_with(str(record.msg), secrets) if record.msg is not None else record.msg
        record.args = self._scrub_args(record.args, secrets)
        if record.exc_info and not record.exc_text:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = _scrub_with(record.exc_text, secrets)
        if record.stack_info:
            record.stack_info = _scrub_with(record.stack_info, secrets)
        return True

    @staticmethod
    def _scrub_args(
        args: tuple[object, ...] | Mapping[str, object] | None,
        secrets: set[str],
    ) -> tuple[object, ...] | Mapping[str, object] | None:
        """Scrub arguments without changing their shape.

        A lone Mapping must stay a Mapping: logging unwraps it into ``args``,
        and turning it into a tuple of keys makes ``getMessage()`` raise
        "format requires a mapping". Keys are coerced to ``str`` because
        ``%(name)s``-style formatting requires string keys regardless of what
        the caller passed in.
        """
        if not args:
            return args
        if isinstance(args, Mapping):
            return {str(key): _scrub_with(str(value), secrets) for key, value in args.items()}
        return tuple(
            _scrub_with(str(arg), secrets) if not isinstance(arg, (int, float)) else arg
            for arg in args
        )


def install_secret_filter() -> int:
    """Attach ``SecretFilter`` to every output handler, idempotently.

    Handlers, not loggers: a logger's filters run only for records that
    originate on it, and propagation to an ancestor runs the ancestor's
    *handlers*, not its *filters*. This plugin logs through at least ten loggers
    across two hierarchies (``supybot.plugins.LLM.*`` and ``llm.verse.*``), so
    per-logger installation would cover whichever two we happened to name.

    In this deployment, only the ``supybot`` hierarchy actually has a handler
    attached anywhere (supybot's own startup does that); nothing in this repo
    attaches one to root or to ``llm`` (``grep addHandler plugins/llm/src/`` —
    no hits). So an ``llm.verse.*`` record propagates all the way up without
    finding a single handler, and Python's fallback path takes over:
    ``logging.lastResort`` — a bare stderr handler that belongs to no logger —
    handles it instead, unfiltered, straight into ``docker logs``. Filtering
    ``logging.lastResort`` here closes that path regardless of whether a real
    handler ever gets attached to ``llm`` later.

    Handlers created later — supybot adds per-plugin file handlers when
    ``individualLogfiles`` is true; prod has it false — are not covered. Calling
    this again picks them up.

    Returns the number of handlers newly filtered, for the startup log line.
    """
    installed = 0
    targets = [
        logging.getLogger(),
        logging.getLogger("supybot"),
        logging.getLogger("llm"),
    ]
    handlers = [handler for logger in targets for handler in logger.handlers]
    if logging.lastResort is not None:
        handlers.append(logging.lastResort)
    for handler in handlers:
        if not any(isinstance(existing, SecretFilter) for existing in handler.filters):
            handler.addFilter(SecretFilter())
            installed += 1
    return installed
