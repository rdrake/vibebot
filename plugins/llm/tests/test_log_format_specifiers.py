"""Supybot log calls must not use %d.

``supybot.log.Logger._log`` routes every argument through
``supybot.utils.str.format``, a mini-language that is NOT printf. It supports
``%s``, ``%r``, ``%i``, ``%f``, ``%.3f`` and ``%%`` -- and has no ``%d``. An
unsupported ``%d`` is left in the output literally and the positional args
shift left into whichever slots are supported. No exception, no warning, just
a wrong line:

    log.info("redaction: %d handler(s), %d var(s): %s", 4, 2, "A, B")
    -> "redaction: %d handler(s), %d var(s): 4"

That is a real line this plugin printed in production. It is not confined to
supybot's own loggers either: ``supybot.log`` calls ``logging.setLoggerClass``
at import, so every logger in the process is affected, including plain
``logging.getLogger("llm.verse.store")``.

32 log lines carried this defect before the sweep that added this test. It
costs nothing to keep them fixed, and the failure is invisible in review --
``%d`` in a log call looks completely ordinary.
"""

from __future__ import annotations

import ast
from pathlib import Path

LOG_METHODS = {"debug", "info", "warning", "error", "exception", "critical"}
SRC_ROOT = Path(__file__).resolve().parents[3]


def _log_format_strings() -> list[tuple[Path, int, str]]:
    """Every literal format string passed to a logging call under plugins/."""
    found: list[tuple[Path, int, str]] = []
    for path in sorted(SRC_ROOT.glob("plugins/*/src/**/*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in LOG_METHODS
                and node.args
            ):
                continue
            fmt = node.args[0]
            if isinstance(fmt, ast.Constant) and isinstance(fmt.value, str):
                found.append((path, node.lineno, fmt.value))
    return found


def test_the_scan_actually_finds_log_calls() -> None:
    """Guards the guard: a broken scanner would make the real test vacuous."""
    assert len(_log_format_strings()) > 100


def test_no_log_call_uses_percent_d() -> None:
    """%d silently corrupts the line. Use %i."""
    offenders = [
        f"{path.relative_to(SRC_ROOT)}:{lineno}: {fmt[:70]}"
        for path, lineno, fmt in _log_format_strings()
        if "%d" in fmt
    ]
    assert not offenders, "supybot's format() has no %d -- use %i:\n" + "\n".join(offenders)


def test_percent_d_really_is_broken() -> None:
    """Pins the defect itself, so this test file explains why it exists.

    If supybot ever gains %d this fails, and the rule above can be dropped.
    """
    from supybot.utils.str import format as supy_format

    assert supy_format("%d", 3) == "%d"
    # ...and the arg shifts into the next supported slot rather than vanishing.
    assert supy_format("%d %s", 1, 2) == "%d 1"
