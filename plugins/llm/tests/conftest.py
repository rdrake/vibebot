"""Pytest configuration and shared fixtures for LLM plugin tests."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import socket
import threading
import time
from collections.abc import Callable, Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest
from llm import provider
from llm.identity import Caller, Identity
from llm.persistence import LLMDatabase, ReminderRow
from llm.plugin import LLM
from llm.service import LLMService
from llm.typing_holds import TypingHolds
from supybot import ircutils

if TYPE_CHECKING:
    from unittest.mock import Mock

    from pytest_mock import MockerFixture

# =============================================================================
# Test constants
# =============================================================================

TEST_MODEL = "xai/grok-4.3"
TEST_URL_BASE = "https://example.com/llm"

# Fake values, each comfortably over apikeys.MIN_REDACTABLE_LEN.
FAKE_PROVIDER_KEYS = {
    "XAI_API_KEY": "xai-fake-key-for-tests-0000",
    "GEMINI_API_KEY": "AIza-fake-key-for-tests-0000",
    "OPENAI_API_KEY": "sk-fake-key-for-tests-0000",
    "ANTHROPIC_API_KEY": "sk-ant-fake-key-for-tests-0000",
}

# Duplicated from llm.apikeys (Task 2) rather than imported: that module
# does not exist yet and this fixture must not depend on it. Task 3 adds a
# test asserting the two copies stay in sync.
_SECRET_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET", "_CREDENTIALS")

# =============================================================================
# Autouse fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _isolate_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every test a known, fake set of provider credentials.

    A developer's shell may carry real keys. Without this, tests can pass
    locally against real keys, fail in CI, print real keys into a failure
    diff, or reach the network.
    """
    for name in list(os.environ):
        if name.upper().endswith(_SECRET_SUFFIXES):
            monkeypatch.delenv(name, raising=False)
    for name, value in FAKE_PROVIDER_KEYS.items():
        monkeypatch.setenv(name, value)


@pytest.fixture(autouse=True)
def _no_ctcp_time_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop a reminder test from waiting on a client that will never answer.

    Any fixture that builds a real LLM and schedules a reminder runs the CTCP
    TIME probe; at the production timeout each one sleeps 3 seconds and skews
    tests that compare fire times against time.time().
    """
    monkeypatch.setattr(LLM, "_CTCP_TIME_TIMEOUT_SECONDS", 0.0)


@pytest.fixture(autouse=True)
def _restore_global_logging_filters() -> Generator[None]:
    """Undo any ``SecretFilter`` installation left on process-global logging state.

    ``apikeys.install_secret_filter()`` (installed from ``LLM.__init__`` as of
    Task 4) attaches to handlers already sitting on root/``supybot``/``llm``,
    plus
    ``logging.lastResort`` — real, process-global state, not anything scoped
    to a single test. Any test that constructs a real plugin (``LLM(mock_irc)``,
    not just ``make_service()``) triggers this via ``__init__``. pytest reuses
    one session-scoped ``LogCaptureHandler`` for the whole run, so a filter
    installed by one test survives into every later test's ``caplog`` —
    silently turning any of ``FAKE_PROVIDER_KEYS``'s values into
    ``[REDACTED]`` in unrelated assertions, well after the test that
    installed the filter has finished. Snapshotting and restoring
    ``handler.filters`` here, for every test in the suite, confines
    installation to the test that triggered it.

    Generalized from the equivalent class-local fixture this module used to
    carry only for ``TestInstallSecretFilter`` in ``test_apikeys.py`` — every
    test that builds a real plugin needs the same protection, not just the
    tests that exercise ``install_secret_filter`` directly.
    """
    loggers = [logging.getLogger(name) for name in ("", "supybot", "llm")]
    handler_snapshot = {
        handler: list(handler.filters) for logger in loggers for handler in logger.handlers
    }
    last_resort_snapshot = (
        list(logging.lastResort.filters) if logging.lastResort is not None else None
    )
    yield
    for handler, filters in handler_snapshot.items():
        handler.filters = filters
    if logging.lastResort is not None and last_resort_snapshot is not None:
        logging.lastResort.filters = last_resort_snapshot


@pytest.fixture(autouse=True)
def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse outbound network, but permit loopback.

    The guard exists to stop tests reaching real providers. A test-local
    HTTP server on 127.0.0.1 is not a provider call, and blocking it would
    make the redirect-refusal tests in test_statuspage_fetch.py impossible
    to write without duplicating this fixture.
    """
    loopback = {"127.0.0.1", "::1", "localhost"}
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def _guard(real):
        def _inner(self, addr, *args, **kwargs):
            host = addr[0] if isinstance(addr, tuple) else None
            if host in loopback:
                return real(self, addr, *args, **kwargs)
            raise RuntimeError("test attempted a real network connection — mock the provider call")

        return _inner

    monkeypatch.setattr(socket.socket, "connect", _guard(real_connect))
    monkeypatch.setattr(socket.socket, "connect_ex", _guard(real_connect_ex))


@pytest.fixture(autouse=True)
def _run_dispatch_threads_inline(mocker: MockerFixture) -> None:
    """Run ``_dispatch_addressed_async`` inline instead of on a daemon thread.

    Production offloads addressed-message dispatch (doPrivmsg / invalidCommand)
    to a ``world.SupyThread`` so the IRC driver thread is freed to flush the
    ``+typing`` indicator immediately instead of blocking on LLM generation
    (otherwise "is composing" only appears at the same instant as the reply).
    Tests assert on dispatch side effects right after the call, so replacing
    ``SupyThread`` with an inline runner avoids a worker-thread race. Tests
    that need to inspect the offload itself re-patch ``SupyThread`` locally.
    """

    class _InlineThread:
        def __init__(self, *args: object, target: object = None, **kwargs: object) -> None:
            self._target = target

        def start(self) -> None:
            if callable(self._target):
                self._target()

    mocker.patch("llm.plugin.world.SupyThread", _InlineThread)


# Daemon threads that only ``LLM.die()`` stops: the animate render refresher
# started from ``LLM.__init__`` and the shared ``TypingHolds`` keepalive that
# ``LLMService`` starts lazily on the first hold.
PLUGIN_LIFECYCLE_THREADS = ("animate-typing-refresher", "typing-keepalive")


@contextlib.contextmanager
def reaping_undead_plugins() -> Generator[None]:
    """``die()`` every ``LLM`` built inside the block that nobody else died.

    ``LLM.__init__`` starts the ``animate-typing-refresher`` daemon thread, and
    ``LLMService`` owns a ``TypingHolds`` registry whose ``typing-keepalive``
    thread starts on the first hold. Only ``die()`` stops either. The
    ``plugin_env`` fixture does that in its ``finally``, but ~150 tests build a
    plugin some other way — a bare ``LLM(mock_irc)`` in the test body, or a
    module-local fixture — and simply drop it. Each abandoned plugin parked a
    thread, plus the plugin object and its whole mock graph, for the rest of
    the session: a full run used to end with 131 live refresher threads.
    Nothing in production leaks this way (Limnoria always calls ``die()``), so
    the fix belongs here rather than in ``plugin.py``.

    Patching ``__init__`` registers each instance; patching ``die`` unregisters
    it, so a plugin the test already died is not died a second time. ``die()``
    is written to be idempotent and this fixture deliberately does not lean on
    that — a regression in its idempotence should surface in the tests that
    exercise it, not be papered over here.

    Both attributes are restored before any reaping runs, so a nested block
    (the fixture below plus a test using this helper directly) hands the
    ``die()`` back to the enclosing block's tracker instead of re-entering
    its own.
    """
    live: dict[int, LLM] = {}
    real_init = LLM.__init__
    real_die = LLM.die

    def tracking_init(self: LLM, irc: Any) -> None:
        real_init(self, irc)
        live[id(self)] = self

    def tracking_die(self: LLM) -> None:
        live.pop(id(self), None)
        real_die(self)

    LLM.__init__ = tracking_init
    LLM.die = tracking_die
    try:
        yield
    finally:
        LLM.__init__ = real_init
        LLM.die = real_die
        while live:
            _, plugin = live.popitem()
            plugin.die()


@pytest.fixture(autouse=True)
def _reap_undead_plugins(mocker: MockerFixture) -> Generator[None]:
    """Apply :func:`reaping_undead_plugins` to every test in the suite.

    ``mocker`` is requested purely for teardown ordering: a fixture is torn
    down before the fixtures it depends on, so this reaps while
    ``plugin_init_patches``'s patches are still installed and ``die()`` still
    meets the mocked ``schedule``/``httpserver``/``LLMDatabase`` the test built
    the plugin against. Reaping after pytest-mock's undo would send a dying
    test plugin at the real Limnoria scheduler.
    """
    with reaping_undead_plugins():
        yield


# =============================================================================
# Session-scoped fixtures
# =============================================================================


@pytest.fixture(scope="session", autouse=True)
def assert_no_plugin_threads_survive_session() -> Generator[None]:
    """Fail the run if a plugin lifecycle thread outlives the whole session.

    The per-test reaper above is only as good as its coverage; this is the
    assertion that says so. Function-scoped teardown has finished for every
    test by the time a session-scoped finalizer runs, so any thread still
    standing here belongs to a plugin nobody died — the exact leak the reaper
    exists to close.
    """
    yield
    survivors = sorted(t.name for t in threading.enumerate() if t.name in PLUGIN_LIFECYCLE_THREADS)
    assert not survivors, (
        f"{len(survivors)} plugin lifecycle thread(s) outlived the session: "
        f"{survivors}. Some test built an LLM the reaper did not see."
    )


@pytest.fixture(scope="session", autouse=True)
def cleanup_limnoria_logging() -> Generator[None]:
    """Clean up Limnoria's logging handlers to prevent errors on shutdown.

    Limnoria registers atexit handlers that try to log shutdown messages.
    When pytest closes stdout before these handlers run, it causes
    'I/O operation on closed file' errors. This fixture removes
    Limnoria's stream handlers before pytest cleanup.
    """
    yield

    # Remove all stream handlers from supybot logger to prevent
    # logging to closed stdout/stderr after pytest finishes
    try:
        import supybot.log

        supybot_logger = logging.getLogger("supybot")
        for handler in supybot_logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler):
                supybot_logger.removeHandler(handler)

        # Also clean the root supybot module logger if it exists
        if hasattr(supybot.log, "log"):
            for handler in supybot.log.log.handlers[:]:
                if isinstance(handler, logging.StreamHandler):
                    supybot.log.log.removeHandler(handler)
    except (ImportError, AttributeError):
        pass

    # Unregister supybot's atexit handlers that cause the logging errors
    try:
        import supybot.world

        # Clear the atexit callbacks that supybot registers
        if hasattr(supybot.world, "dying"):
            supybot.world.dying = True  # Prevent further shutdown logging
    except (ImportError, AttributeError):
        pass


# =============================================================================
# Database fixture
# =============================================================================


@pytest.fixture
def test_db(tmp_path: Path) -> Generator[LLMDatabase, None, None]:
    """Create an LLMDatabase backed by a temporary file with automatic cleanup."""
    db = LLMDatabase(str(tmp_path / "test.db"))
    yield db
    db.close()


# =============================================================================
# Reminder fixture helpers
# =============================================================================


def make_reminder_row(
    *,
    event_name: str = "evt",
    nick: str = "testnick",
    channel: str = "#test",
    message: str = "",
    action_prompt: str = "",
    account: str | None = None,
    fire_at: float = 0.0,
    chain_position: int = 1,
    recurrence_seconds: int | None = None,
    recurrence_rrule: str | None = None,
    watch_mode: bool = False,
    id: int = 0,  # noqa: A002 — keyword-only builder, builtin shadow is fine.
    created_at: float = 0.0,
    reply_msgid: str = "",
) -> ReminderRow:
    """Build a ReminderRow with sensible defaults for tests.

    Localizes test exposure to the ReminderRow shape so future column churn
    only touches this helper. Keyword-only on purpose: positional args invite
    silent breakage when the row layout changes.
    """
    return ReminderRow(
        id=id,
        event_name=event_name,
        nick=nick,
        channel=channel,
        message=message,
        action_prompt=action_prompt,
        account=account,
        fire_at=fire_at,
        created_at=created_at,
        chain_position=chain_position,
        recurrence_seconds=recurrence_seconds,
        recurrence_rrule=recurrence_rrule,
        watch_mode=watch_mode,
        reply_msgid=reply_msgid,
    )


# The database calls the reminder commands and the agenda make. A plugin
# fixture with a mocked database routes these to a real SQLite store, so
# reminder state lives where it does in production (the row is the only
# truth) while the mock still records every call.
_REMINDER_STORE_CALLS = (
    "save_reminder",
    "get_reminder",
    "delete_reminder",
    "load_pending_reminders",
    "load_due_reminders",
    "reminder_fire_times",
    "update_reminder_fire_at",
)


def back_reminders_with_sqlite(db_mock: MagicMock, path: Path) -> LLMDatabase:
    """Route ``db_mock``'s reminder calls to a real database at ``path``."""
    store = LLMDatabase(str(path))
    for name in _REMINDER_STORE_CALLS:
        getattr(db_mock, name).side_effect = getattr(store, name)
    db_mock.sqlite_reminders = store
    return store


def put_reminder(plugin: Any, row: ReminderRow) -> None:
    """Store ``row`` as a pending reminder, without recording a save call.

    A ``fire_at`` of 0 (the builder's default) means "an hour from now";
    a non-zero ``created_at`` is kept, for tests of the amend window.
    """
    store: LLMDatabase = plugin.db.sqlite_reminders
    store.save_reminder(
        row.event_name,
        row.nick,
        row.channel,
        row.message,
        row.fire_at or time.time() + 3600,
        action_prompt=row.action_prompt,
        account=row.account,
        chain_position=row.chain_position,
        recurrence_seconds=row.recurrence_seconds,
        recurrence_rrule=row.recurrence_rrule,
        watch_mode=row.watch_mode,
        reply_msgid=row.reply_msgid,
    )
    if row.created_at:
        with store._write_txn() as conn:
            conn.execute(
                "UPDATE reminders SET created_at = ? WHERE event_name = ?",
                (row.created_at, row.event_name),
            )


def due_reminder(
    plugin: Any, nick: str, channel: str, message: str, event_name: str, **fields: Any
) -> Callable[[], None]:
    """Store a reminder that is due now. Calling the result fires it, as the
    agenda wakeup does: claim, deliver, and settle the row."""
    plugin.db.sqlite_reminders.delete_reminder(event_name)
    put_reminder(
        plugin,
        make_reminder_row(
            event_name=event_name,
            nick=nick,
            channel=channel,
            message=message,
            fire_at=time.time() - 1,
            **fields,
        ),
    )
    return plugin._agenda_tick


def reminder_names(plugin: Any) -> set[str]:
    """Event names of every stored reminder."""
    store: LLMDatabase = plugin.db.sqlite_reminders
    return {name for name, _at in store.reminder_fire_times()}


# =============================================================================
# provider reply builders
# =============================================================================


def make_caller(
    irc: Any,
    msg: Any,
    nick: str = "testnick",
    *,
    account: str | None = None,
    channel: str | None = None,
) -> Caller:
    """Point ``msg`` and ``irc`` at one consistent sender; return its Caller.

    Sets the nick, prefix, args, channel and IRCv3 account tag on ``msg``, and
    makes ``irc.state.nickToAccount`` agree, so the plugin resolves exactly
    the returned Caller. ``channel=None`` is a private message: ``args[0]``
    becomes the bot's own nick, as it is on the wire. The message text in
    ``args[1]`` is kept.
    """
    args = msg.args if isinstance(getattr(msg, "args", None), tuple) else ()
    text = args[1] if len(args) > 1 else ""
    msg.nick = nick
    msg.prefix = f"{nick}!user@host"
    msg.args = (channel or irc.nick, text)
    msg.channel = channel
    msg.server_tags = {"account": account} if account else {}
    irc.state.nickToAccount = MagicMock(
        side_effect=lambda n: account if ircutils.strEqual(n, nick) else None
    )
    return Caller(Identity(nick, account), channel)


def make_tool_call(
    name: str,
    arguments: dict[str, Any] | str | None = None,
    *,
    call_id: str = "call_0",
) -> provider.ToolCall:
    """Build a tool call as provider.chat returns it.

    ``arguments`` may be a dict (JSON-encoded for you) or a raw string (passed
    through verbatim — useful for exercising malformed-JSON handling).
    """
    if arguments is None:
        arguments = {}
    arguments_str = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return provider.ToolCall(id=call_id, name=name, arguments=arguments_str)


def make_completion_response(
    content: str | None = "response text",
    *,
    tool_calls: list[provider.ToolCall] | None = None,
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
    finish_reason: str = "stop",
    cost: float = 0.0,
    thinking: tuple[dict[str, Any], ...] = (),
) -> provider.Reply:
    """Build a provider.Reply, the one shape every chat call returns."""
    return provider.Reply(
        text=content or "",
        tool_calls=tuple(tool_calls or ()),
        finish_reason=finish_reason,
        usage=provider.Usage(input_tokens=prompt_tokens, output_tokens=completion_tokens),
        cost=cost,
        thinking=thinking,
    )


# =============================================================================
# Shared fixtures for plugin initialization
# =============================================================================


@pytest.fixture
def mock_irc(mocker: MockerFixture) -> MagicMock:
    """Create a mock IRC object suitable for plugin initialization.

    Provides the minimum attributes required by LLM.__init__:
    nick, state.channels, and state.capabilities_ack.
    """
    irc = mocker.MagicMock()
    irc.nick = "testbot"
    irc.state = mocker.MagicMock()
    irc.state.channels = {}
    irc.state.capabilities_ack = set()
    # Default: no NickServ account (nick fallback)
    irc.state.nickToAccount = mocker.MagicMock(return_value=None)
    return irc


@pytest.fixture
def plugin_env(mocker: MockerFixture, tmp_path: Path):
    """Create an LLM plugin instance wired to mocked dependencies.

    Returns (plugin, mock_irc, mock_msg) ready for command invocation.
    """
    registry = make_registry_side_effect()

    mock_irc = mocker.MagicMock()
    mock_irc.nick = "testbot"
    mock_irc.state = mocker.MagicMock()
    mock_irc.state.channels = {"#test": mocker.MagicMock(topic="Test topic")}
    mock_irc.state.capabilities_ack = set()
    # Default: no NickServ account (nick fallback)
    mock_irc.state.nickToAccount = mocker.MagicMock(return_value=None)

    mock_msg = mocker.MagicMock()
    mock_msg.prefix = "testnick!user@host"
    mock_msg.args = ("#test", "test message")
    mock_msg.time = time.time() + 100  # future time -- not ZNC playback
    mock_msg.channel = "#test"
    mock_msg.nick = "testnick"
    mock_msg.server_tags = {}  # default: no IRCv3 account-tag

    mocker.patch.object(LLM, "registryValue", side_effect=registry)
    mocker.patch("llm.plugin.LLMService")
    mocker.patch("llm.plugin.LLMDatabase")
    mocker.patch("llm.plugin.log")
    mocker.patch("llm.plugin.httpserver")
    mocker.patch("llm.plugin.schedule.addPeriodicEvent")
    mocker.patch("llm.plugin.schedule.removeEvent")
    mocker.patch("llm.plugin.schedule.addEvent")
    # Default: registered user (grant llm.* command caps but not owner/admin/trusted)
    mocker.patch(
        "llm.plugin.ircdb.checkCapability",
        side_effect=lambda prefix, cap: cap.startswith("llm."),
    )

    plugin = LLM(mock_irc)
    # After __init__, swap registryValue to a plain MagicMock so
    # each test can override specific keys while keeping defaults.
    plugin.registryValue = mocker.MagicMock(side_effect=registry)

    # Provide the MetaSynchronized RLock that _allow_concurrent expects.
    plugin._MetaSynchronized_rlock = threading.RLock()

    # sanitize_output is a passthrough in tests (the mock would return MagicMock).
    plugin.llm_service.sanitize_output.side_effect = lambda x: x
    # Draw and video are available unless a test says otherwise.
    plugin.llm_service.draw_unavailable_reason.return_value = None
    plugin.llm_service.video_available.return_value = True

    # An unconfigured mock database is an EMPTY database, not one holding a
    # MagicMock's worth of queued clips. The animate admission caps compare
    # these against an int, so the bare mock's auto-attribute would raise
    # TypeError in every test that submits a clip without caring about caps.
    plugin.db.count_pending_animate.return_value = 0
    # No @tz set, and no CTCP TIME probe: a real one would wait on a client
    # that is not there. Probe tests call LLM._probe_ctcp_tz directly.
    plugin.db.get_user_timezone.return_value = None
    plugin.db.get_clock_offset.return_value = None
    plugin._probe_ctcp_tz = mocker.MagicMock(return_value=None)
    plugin.db.count_pending_animate_for.return_value = 0
    # Reminder rows live in real SQLite: the row is the only cancel truth.
    back_reminders_with_sqlite(plugin.db, tmp_path / "reminders.db")

    # The typing registry is real even though the service is mocked: the
    # render refresher's whole job is to reconcile a hold group, and a mock
    # would assert only that a call happened. Sends still land on the mocked
    # send_typing_indicator, so tests keep asserting on it. The long interval
    # keeps the keepalive out of the assertions; plugin.die() stops the
    # thread in this fixture's teardown.
    plugin.llm_service.typing = TypingHolds(
        plugin.llm_service.send_typing_indicator, logging.getLogger("test.typing"), interval=60.0
    )

    # No scheduled tasks unless a test seeds them. @remind list/clear iterate
    # this, and the bare mock would hand them a non-iterable MagicMock.
    plugin.llm_service.list_scheduled_llm_tasks.return_value = []

    # During the unified-assistant transition, ask-style command tests still
    # exercise the existing completion bridge through assistant_completion().
    # The bridge wraps CompletionResult into AssistantResult so the chat
    # impl's structured-signal suppression check (last_successful_tool /
    # final_text_after_tools) sees defaults rather than AttributeError.
    def _assistant_completion_bridge(
        prompt: str,
        *,
        nick=None,
        channel=None,
        account=None,
        is_owner=False,
        capabilities=None,
        route_profile=None,
        entry_route="",
        db=None,
        context=None,
        bot_nick=None,
        images=None,
        history=None,
        channel_history=None,
        irc=None,
        msg=None,
        system_prompt=None,
        memories=None,
        user_instruction=None,
        search_fn=None,
        fetch_fn=None,
        code_fn=None,
        draw_fn=None,
        animate_fn=None,
        cleanup_fn=None,
        set_reminder_fn=None,
        list_pending_tasks_fn=None,
        cancel_pending_task_fn=None,
        cancel_all_pending_tasks_fn=None,
        schedule_llm_task_fn=None,
        extra_tools=None,
        extra_handlers=None,
        model_override=None,
        manage_typing=True,
        exclude_tools=frozenset(),
    ):
        from llm.service import AssistantResult as _AssistantResult
        from llm.service import Spend

        completion_result = plugin.llm_service.completion(
            prompt,
            command="ask",
            images=images,
            history=history,
            channel_history=channel_history,
            irc=irc,
            msg=msg,
            system_prompt=system_prompt,
            memories=memories,
            user_instruction=user_instruction,
        )
        return _AssistantResult(
            content=completion_result.content,
            spend=Spend(
                prompt_tokens=completion_result.prompt_tokens,
                completion_tokens=completion_result.completion_tokens,
                cost=completion_result.cost,
            ),
            model=completion_result.model,
            grounding_used=completion_result.grounding_used,
            error=completion_result.error,
        )

    plugin.llm_service.assistant_completion.side_effect = _assistant_completion_bridge

    try:
        yield plugin, mock_irc, mock_msg
    finally:
        # die() must be idempotent under the executor wiring (see Task 3
        # — shutdown is idempotent, db.close is sqlite-idempotent). Don't
        # suppress exceptions here: if die() raises, that's a real
        # lifecycle bug we want the test to surface.
        plugin.die()


def huge_png(width: int, height: int) -> bytes:
    """A valid PNG header claiming ``width`` x ``height``, under 100 bytes.

    The decompression-bomb shape: tiny on the wire, enormous once decoded.
    Only the header is rewritten, so nothing in the test allocates the pixels.
    """
    import io
    import struct
    import zlib

    from PIL import Image

    buf = io.BytesIO()
    Image.new("L", (1, 1)).save(buf, "PNG")
    data = bytearray(buf.getvalue())
    ihdr = struct.pack(">II", width, height) + bytes(data[24:29])
    data[16:29] = ihdr
    data[29:33] = struct.pack(">I", zlib.crc32(b"IHDR" + ihdr))
    return bytes(data)


def make_registry_side_effect(overrides: dict[str, Any] | None = None):
    """Create a registryValue side_effect function with standard defaults.

    The base defaults provide the superset of config keys used across all test
    files. Tests can override specific keys via the ``overrides`` parameter.

    Args:
        overrides: Optional dict of config keys to override or extend.

    Returns:
        A callable suitable for ``registryValue.side_effect``.
    """
    defaults: dict[str, Any] = {
        # Capability-based settings (Phase 2 Task 5a/5b — sole surface).
        "assistantModel": TEST_MODEL,
        "assistantSystemPrompt": "You are helpful.",
        "imageModel": "xai/grok-imagine-image",
        # Plugin-level init config
        "httpRoot": "",
        "databasePath": "",
        "contextMaxMessages": 20,
        "contextTimeoutMinutes": 30,
        "contextEnabled": True,
        "channelContextMaxMessages": 10,
        "privacyOptInOnly": True,
        # Code command
        "codeModel": TEST_MODEL,
        "codeSystemPrompt": "You write code.",
        # Bot-to-bot loop guard
        "botLoopReplyLimit": 3,
        "botLoopWindow": 300,
        "botLoopHostSuffixes": ["Bot.AfterNET.Org"],
        # Aurora (mirrors config.py: tool on, announcements opt-in)
        "auroraEnabled": True,
        "auroraAnnounce": False,
        "auroraAnnouncePlaces": [],
        # Draw command
        "drawTimeout": 60,
        "drawAutoRewriteMax": 2,
        "drawContextMaxAgeSeconds": 60,
        # Animate command. animateApiUrl is empty by default so the feature is
        # OFF unless a test opts in — the same posture as production before an
        # operator points it at a box.
        "animateApiUrl": "",
        "animateModel": "",
        "animateSize": "1280x704",
        "animateDuration": 4,
        "animateSteps": 25,
        "animateAudio": True,
        "animateFlowShift": 12,
        "animateAudioFlowShift": 3,
        "animateTimeout": 60,
        # Admission caps and the wait estimate (mirrors config.py defaults).
        "animateMaxPendingPerUser": 2,
        "animateMaxPending": 6,
        "animateRenderSeconds": 135,
        # Expiry (pending task retry)
        "askExpiry": 60,
        "codeExpiry": 60,
        "drawExpiry": 60,
        "animateExpiry": 1800,
        # Memory extraction
        "memoryEnabled": True,
        "memoryMaxPerUser": 50,
        "memoryCleanupInterval": 3,
        "memoryPromotionThreshold": 2,
        "memoryCandidateTTLDays": 14,
        # Shared
        "timeout": 30,
        "maxPromptLength": 10000,
        "commandPrefixes": ["."],
        "httpUrlBase": TEST_URL_BASE,
        "longReplyTeaserMaxChars": 220,
        "multilineMaxLines": 3,
        "fileCleanupAge": 24,
        "fileCleanupMax": 100,
        "logLevel": "WARNING",
        # Rate limiting
        "enforceRateLimits": True,
        # ask
        "askRateLimitCount": 15,
        "askRateLimitWindow": 60,
        "askTrustedRateLimitCount": 15,
        "askTrustedRateLimitWindow": 60,
        "askUnregRateLimitCount": 15,
        "askUnregRateLimitWindow": 60,
        # code
        "codeRateLimitCount": 10,
        "codeRateLimitWindow": 60,
        "codeTrustedRateLimitCount": 0,
        "codeTrustedRateLimitWindow": 60,
        "codeUnregRateLimitCount": 2,
        "codeUnregRateLimitWindow": 60,
        # draw
        "drawRateLimitCount": 2,
        "drawRateLimitWindow": 300,
        "drawTrustedRateLimitCount": 5,
        "drawTrustedRateLimitWindow": 60,
        "drawUnregRateLimitCount": 0,
        "drawUnregRateLimitWindow": 60,
        # animate (mirrors config.py: stricter than draw on every tier, since a
        # clip is ~70s of exclusive GPU time and the box serialises)
        "animateRateLimitCount": 2,
        "animateRateLimitWindow": 900,
        "animateTrustedRateLimitCount": 4,
        "animateTrustedRateLimitWindow": 300,
        "animateUnregRateLimitCount": 1,
        "animateUnregRateLimitWindow": 7200,
        # meme (mirrors config.py)
        "memeRateLimitCount": 6,
        "memeRateLimitWindow": 60,
        "memeTrustedRateLimitCount": 12,
        "memeTrustedRateLimitWindow": 60,
        "memeUnregRateLimitCount": 2,
        "memeUnregRateLimitWindow": 600,
        "memeAliases": [],
        # Opt-in per test: True here would add make_meme to every extra_tools
        # assertion in the irc_lookup wiring tests.
        "memeEnabled": False,
        # Same reason: opt in per test (config.py defaults it True).
        "reactEnabled": False,
        "redactEnabled": False,
        "notifyOnlineEnabled": False,
        # story (mirrors draw — expensive image command)
        "storyRateLimitCount": 2,
        "storyRateLimitWindow": 300,
        "storyTrustedRateLimitCount": 5,
        "storyTrustedRateLimitWindow": 60,
        "storyUnregRateLimitCount": 0,
        "storyUnregRateLimitWindow": 60,
        # Search/fetch tools
        "searchModel": "",
        # Subject research (the @draw / @animate dossier pre-stage). OFF here,
        # unlike config.py's default of True: it puts a second LLM call in
        # front of both planners, and every existing draw and animate test
        # asserts on the one call the command used to make. Tests that want it
        # opt in with make_registry_side_effect({"subjectResearchEnabled": True}).
        "subjectResearchEnabled": False,
        "subjectResearchModel": "",
        # Assistant tool-calling backend
        "metaMaxSteps": 7,
        # IRCv3 join optimization
        "skipAutoWhoOnJoin": True,
        # Pending-task tool gate (default off, mirrors config.py)
        "pendingTasksEnabled": False,
        # Limnoria bridge
        "bridgeEnabled": False,
        "bridgeAllowedPlugins": [],
        "bridgeAllowMutating": False,
        "bridgeDebugInChannel": False,
        # Async LLM concurrency cap
        "maxConcurrentLLMCalls": 16,
        # Verse subsystem
        "verseEnabled": False,
        "verseCompactionModel": "gemini/gemini-flash-lite-latest",
        "verseStyleExemplars": [],
        # Service status tool — a single configured source by default so
        # _status_sources() (bound below onto the plugin Mock) returns a real
        # list instead of an unconfigured Mock's auto-attribute.
        "statusPageUrls": ["https://status.claude.com"],
        # Queryable-only allowlist (Task 4). Empty by default, matching
        # config.py's default and keeping the "polled OR queryable" gate
        # exercised by TestQueryableOnlyGate opt-in per test rather than
        # incidental via this fixture's own unconfigured-key fallback.
        "statusQueryablePages": [],
        # Feature switches default on, matching config.py.
        "drawEnabled": True,
        "animateEnabled": True,
    }
    if overrides:
        defaults.update(overrides)

    def side_effect(key: str, *args: object) -> object:
        return defaults.get(key, "")

    return side_effect


# =============================================================================
# Service factory fixture
# =============================================================================


@pytest.fixture
def make_service(mocker: MockerFixture) -> Callable[..., tuple[LLMService, Mock]]:
    """Factory fixture that creates an LLMService with standard config defaults.

    Usage::

        def test_something(make_service):
            service, plugin = make_service()
            # or with overrides:
            service, plugin = make_service(assistantModel="gemini/gemini-2.0-flash")

    Every service built here is torn down at the end of the test so its
    TypingHolds keepalive thread does not outlive it.
    """
    built: list[LLMService] = []

    def _make(**overrides: Any) -> tuple[LLMService, Mock]:
        plugin = mocker.Mock()
        plugin.log = mocker.Mock()
        plugin.registryValue = mocker.Mock(side_effect=make_registry_side_effect(overrides or None))

        # Bind the real readers, not stubs returning a fixed list/dict: a
        # prior incident shipped a commit that read a deleted registry key,
        # and a stubbed _status_sources would have hidden that structurally.
        # Routing through the real methods means a mistyped or removed
        # registry key collapses to [] / {} here too, the same as it would
        # in production. _status_named_pages is now the gate service.py
        # actually reads (polled OR queryable), so every caller of
        # assistant_completion needs it wired, not just the status-specific
        # test classes that bind it themselves.
        from llm.plugin import LLM

        plugin._STATUS_MAX_SOURCES = LLM._STATUS_MAX_SOURCES
        plugin._STATUS_MAX_QUERYABLE = LLM._STATUS_MAX_QUERYABLE
        plugin._status_host = LLM._status_host.__get__(plugin)
        # Same trap noted above for the other _status_* readers: an unbound
        # attribute on a bare Mock silently returns a fresh Mock instead of
        # running the real bare-URL name sanitiser.
        plugin._status_bare_name = LLM._status_bare_name.__get__(plugin)
        plugin._status_parse_pages = LLM._status_parse_pages.__get__(plugin)
        plugin._status_polled_pages = LLM._status_polled_pages.__get__(plugin)
        plugin._status_sources = LLM._status_sources.__get__(plugin)
        plugin._status_named_pages = LLM._status_named_pages.__get__(plugin)

        # Service tests dispatch scheduled-task fires synchronously to
        # assert downstream effects; with the executor migration the
        # real fire() submits the worker via plugin._llm_executor.submit.
        # Run the worker inline here so existing tests behave as before.
        plugin._llm_executor.closing = False

        def _sync_submit(_label, fn, *args, **kwargs):
            fn(*args, **kwargs)
            return mocker.Mock()

        plugin._llm_executor.submit.side_effect = _sync_submit

        # _safe_queue is the worker-side wrapper around irc.queueMsg.
        # Existing tests assert directly on irc.queueMsg, so make
        # _safe_queue a passthrough that calls queueMsg.
        def _passthrough_safe_queue(irc, msg):
            irc.queueMsg(msg)
            return True

        plugin._safe_queue.side_effect = _passthrough_safe_queue

        # Every raw-queue send builds its IrcMsg with _safe_privmsg (the
        # safeArgument counterpart to irc.reply's). Use the real staticmethod so
        # tests inspecting queueMsg see an actual PRIVMSG, not a bare Mock.
        plugin._safe_privmsg.side_effect = LLM._safe_privmsg

        service = LLMService(plugin)
        built.append(service)
        return service, plugin

    yield _make

    for service in built:
        service.typing.stop()


# =============================================================================
# Plugin initialization patches
# =============================================================================


def plugin_init_patches(
    mocker: MockerFixture, *, mock_database: bool = True
) -> dict[str, MagicMock]:
    """Patch all external dependencies for LLM.__init__.

    This patches LLMService, log, httpserver.hook, schedule.addPeriodicEvent,
    and schedule.removeEvent. Optionally patches LLMDatabase.  All patches are
    automatically reverted at the end of the test by pytest-mock.

    Also stubs LLMExecutor with a synchronous, deterministic test double so
    background-work tests (reminder action, scheduled task)
    can assert on side effects immediately after the schedule callback
    fires — no thread-pool race.

    Args:
        mocker: The pytest-mock fixture.
        mock_database: If True (default), also patches LLMDatabase.

    Returns:
        Dict of patch names to their MagicMock objects.
    """
    patches: dict[str, Any] = {}
    patches["LLMService"] = mocker.patch("llm.plugin.LLMService")
    if mock_database:
        patches["LLMDatabase"] = mocker.patch("llm.plugin.LLMDatabase")
        # Never probed: a bare MagicMock row would be compared with a float.
        patches["LLMDatabase"].return_value.get_clock_offset.return_value = None
    patches["log"] = mocker.patch("llm.plugin.log")
    patches["httpserver_hook"] = mocker.patch("llm.plugin.httpserver.hook")
    patches["addPeriodicEvent"] = mocker.patch("llm.plugin.schedule.addPeriodicEvent")
    patches["removeEvent"] = mocker.patch("llm.plugin.schedule.removeEvent")
    # Also patch addEvent so the daily compaction timer (PR 3 / E3)
    # doesn't hit the real supybot schedule between tests — repeated
    # registrations would otherwise collide on the unique-name check.
    patches["addEvent"] = mocker.patch("llm.plugin.schedule.addEvent")

    # LLMExecutor stub: submit runs the function synchronously so tests
    # that callback() and immediately assert on queueMsg / db writes
    # don't race a worker thread.
    sync_stub = _SyncLLMExecutor()
    patches["LLMExecutor"] = mocker.patch("llm.plugin.LLMExecutor", return_value=sync_stub)
    return patches


class _SyncLLMExecutor:
    """Synchronous test double for LLMExecutor used by plugin_init_patches.

    `submit(label, fn, ...)` runs `fn` inline and returns a completed
    `Future` carrying its result. Counters track in-flight tasks so
    `running()` and `queued()` look correct mid-call. Drain/shutdown
    are no-ops.
    """

    def __init__(self, max_concurrency: int = 16) -> None:
        from concurrent.futures import Future

        self._max = max_concurrency
        self._running = 0
        self._queued = 0
        self.closing = False
        self._Future = Future

    @property
    def max_concurrency(self) -> int:
        return self._max

    def running(self) -> int:
        return self._running

    def queued(self) -> int:
        return self._queued

    def permit(self):
        from contextlib import contextmanager

        @contextmanager
        def _cm():
            self._running += 1
            try:
                yield
            finally:
                self._running -= 1

        return _cm()

    def submit(self, _label, fn, *args, **kwargs):
        fut = self._Future()
        self._running += 1
        try:
            result = fn(*args, **kwargs)
        except BaseException as e:  # noqa: BLE001
            fut.set_exception(e)
        else:
            fut.set_result(result)
        finally:
            self._running -= 1
        return fut

    def drain(self, timeout: float) -> bool:  # noqa: ARG002
        return True

    def shutdown(self) -> None:
        self.closing = True


@pytest.fixture
def status_plugin():
    """A minimal stand-in exercising the status poller logic in isolation.

    Builds the real methods onto a bare object rather than constructing the
    whole LLM plugin, which needs an IRC connection, a database, and an
    executor pool. The methods under test only touch attributes defined here.
    """
    from unittest.mock import MagicMock

    from llm import statuspage
    from llm.plugin import LLM

    obj = MagicMock()
    obj._registry = {"statusPageUrls": ["Claude=https://status.claude.com"]}
    obj._registry["statusQueryablePages"] = []
    obj.registryValue = lambda key, *a, **k: obj._registry.get(key)
    obj._STATUS_POLL_INTERVAL = LLM._STATUS_POLL_INTERVAL
    obj._STATUS_MAX_ANNOUNCE_PER_POLL = LLM._STATUS_MAX_ANNOUNCE_PER_POLL
    obj._STATUS_FETCH_FLOOR = LLM._STATUS_FETCH_FLOOR
    obj._STATUS_MAX_SOURCES = LLM._STATUS_MAX_SOURCES
    obj._STATUS_MAX_QUERYABLE = LLM._STATUS_MAX_QUERYABLE
    obj._status_parse_pages = LLM._status_parse_pages.__get__(obj)
    obj._status_polled_pages = LLM._status_polled_pages.__get__(obj)
    obj._status_named_pages = LLM._status_named_pages.__get__(obj)
    obj._status_sources = LLM._status_sources.__get__(obj)
    # Same trap as every other _status_* helper on this bare MagicMock: an
    # unbound attribute access silently returns a fresh Mock instead of
    # running the real method, which would make the bare-URL name-derivation
    # tests pass vacuously against a Mock rather than the real sanitiser.
    obj._status_bare_name = LLM._status_bare_name.__get__(obj)
    obj._status_state = {}
    obj._status_read_cache = {}
    obj._status_last_fetch = {}
    obj._status_history_cache = {}
    obj._status_history_at = {}
    obj._status_history_failed_at = {}
    obj._status_query_cache = {}
    obj._status_query_failed_at = {}
    obj._STATUS_QUERY_TTL = LLM._STATUS_QUERY_TTL
    obj._STATUS_QUERY_CACHE_MAX = LLM._STATUS_QUERY_CACHE_MAX
    # _status_query_snapshot calls self._status_evict_query_cache() — on a bare
    # MagicMock an unbound attribute access silently returns a fresh Mock
    # instead of running the real method, so eviction would be a no-op unless
    # this is bound explicitly too (same trap as the other _status_* helpers
    # bound below).
    obj._status_evict_query_cache = LLM._status_evict_query_cache.__get__(obj)
    obj._fetch_calls = 0
    obj._fake_snapshot = None
    obj._fake_error = None
    obj._now = 1000.0
    obj._mono = 0.0
    obj._status_monotonic = lambda: obj._mono
    obj._STATUS_MIN_FETCH_WINDOW = LLM._STATUS_MIN_FETCH_WINDOW
    obj._STATUS_HISTORY_TTL = LLM._STATUS_HISTORY_TTL
    obj._STATUS_HISTORY_LIMIT = LLM._STATUS_HISTORY_LIMIT
    obj._STATUS_HISTORY_RETRY = LLM._STATUS_HISTORY_RETRY
    obj._STATUS_PASS_BUDGET = LLM._STATUS_PASS_BUDGET
    obj._STATUS_REWRITE_RESERVE = LLM._STATUS_REWRITE_RESERVE
    obj._STATUS_MAX_LINES_PER_POLL = LLM._STATUS_MAX_LINES_PER_POLL
    obj._STATUS_TOOL_BUDGET = LLM._STATUS_TOOL_BUDGET
    obj._STATUS_STALE_AFTER = LLM._STATUS_STALE_AFTER
    obj._status_cursor = None
    obj._status_rotate = LLM._status_rotate.__get__(obj)
    obj._poll_one_source = LLM._poll_one_source.__get__(obj)

    def fake_fetch(source, *, timeout_cap=None, cached=None):
        obj._fetch_calls += 1
        obj._fetch_sources.append(source)
        if obj._fake_error:
            err, obj._fake_error = obj._fake_error, None
            raise err
        snap = obj._fake_snapshot
        if snap is None:
            snap = statuspage.Snapshot(
                page_name="Claude",
                page_url="https://status.claude.com",
                indicator="none",
                description="All Systems Operational",
                components={},
                incidents={},
                fetched_at=obj._now,
            )
        return snap

    obj._fetch_sources = []
    obj._status_fetch_snapshot = fake_fetch
    obj._status_now = lambda: obj._now
    obj._announce_status = MagicMock()
    obj._run_status_poll = LLM._run_status_poll.__get__(obj)
    obj._status_fetch_now = LLM._status_fetch_now.__get__(obj)
    obj._status_prune_sources = LLM._status_prune_sources.__get__(obj)
    obj._status_host = LLM._status_host.__get__(obj)
    # Helpers _status_tool_payload delegates to for per-source entry
    # construction. Same trap as everything else on this bare MagicMock: an
    # unbound attribute silently returns a fresh Mock instead of running the
    # real method, which would make budget/staleness logic in the payload
    # tests silently no-op rather than fail loudly.
    obj._status_source_entry = LLM._status_source_entry.__get__(obj)
    obj._status_single_payload = LLM._status_single_payload.__get__(obj)

    # Arming/dedup tests bind the real _schedule_status_poll /
    # _enqueue_status_poll. Both traps below matter: `closing` on a bare
    # MagicMock is a truthy Mock (the guard would always fire), and the
    # inflight flag must be a real Event for is_set()/set() to behave.
    obj._llm_executor.closing = False
    obj._status_poll_inflight = threading.Event()
    obj._schedule_status_poll = LLM._schedule_status_poll.__get__(obj)
    obj._enqueue_status_poll = LLM._enqueue_status_poll.__get__(obj)

    # For tests that bind the real _status_fetch_snapshot directly (the
    # 304/replace branch): a numeric timeout and a resolves-public stub.
    obj._registry["timeout"] = 30
    obj.llm_service._resolves_to_public = MagicMock(return_value=True)

    return obj


@pytest.fixture
def spaceweather_plugin():
    """A bare stand-in for the storm announcer, like ``status_plugin``.

    Every real method under test is bound explicitly: on a bare MagicMock an
    unbound attribute returns a fresh Mock instead of running the real code,
    so a forgotten binding passes vacuously. Collaborators that would touch
    the network or IRC are plain callables the tests swap:

    - ``obj._places``: label -> Place, None (no match) or an exception to raise
    - ``obj._samples``: list of Samples, or an exception to raise
    - ``obj._queue_ok``: channel -> bool, what ``_safe_queue`` returns
    - ``obj.sent``: (channel, line) for every queued line
    """
    from datetime import UTC, datetime

    from llm import spaceweather

    obj = MagicMock()
    obj._registry = {}
    obj._chan_registry = {}
    obj.registryValue = lambda key, channel=None, *a, **k: obj._chan_registry.get(
        (key, channel), obj._registry.get(key)
    )
    obj._channels = set()
    obj._all_known_channels = lambda: set(obj._channels)
    obj._now = datetime(2026, 10, 11, 12, 0, tzinfo=UTC)
    obj._sw_now = lambda: obj._now
    obj._mono = 0.0
    obj._status_monotonic = lambda: obj._mono
    obj._status_now = lambda: 1000.0

    obj._places = {}

    def fake_place(query):
        value = obj._places.get(query)
        if isinstance(value, BaseException):
            raise value
        return value

    obj._sw_place = fake_place
    obj._samples = []

    def fake_samples(_now):
        if isinstance(obj._samples, BaseException):
            raise obj._samples
        return obj._samples

    obj._sw_samples = fake_samples

    obj.sent = []
    obj._queue_ok = {}
    obj._irc_for_channel = lambda channel: f"irc:{channel}"
    obj._safe_privmsg = lambda channel, line: (channel, line)

    def fake_queue(_irc, msg):
        ok = obj._queue_ok.get(msg[0], True)
        if ok:
            obj.sent.append(msg)
        return ok

    obj._safe_queue = fake_queue

    obj._sw_lock = threading.Lock()
    obj._sw_cache = {}
    obj._sw_state = spaceweather.SwState()
    obj._sw_place_warned = set()
    obj._sw_poll_inflight = threading.Event()
    obj._llm_executor.closing = False
    obj._SPACEWEATHER_POLL_INTERVAL = LLM._SPACEWEATHER_POLL_INTERVAL
    obj._SPACEWEATHER_PASS_BUDGET = LLM._SPACEWEATHER_PASS_BUDGET
    obj._SW_STORM_CLOUD_HOURS = LLM._SW_STORM_CLOUD_HOURS
    obj._SW_STORM_RESUME_WINDOW = LLM._SW_STORM_RESUME_WINDOW
    obj._SW_BZ_THRESHOLD_NT = LLM._SW_BZ_THRESHOLD_NT
    obj._SW_BZ_HOURS = LLM._SW_BZ_HOURS
    obj._SW_MIN_WATCH_LEVEL = LLM._SW_MIN_WATCH_LEVEL
    obj._SW_ALERTS_TTL = LLM._SW_ALERTS_TTL
    obj._SW_CLOUDS_TTL = LLM._SW_CLOUDS_TTL
    obj._sw_dark_hours = LLM._sw_dark_hours  # staticmethod: no binding
    for name in (
        "_sw_cached",
        "_sw_swpc",
        "_sw_alerts",
        "_sw_cloud_rows",
        "_sw_cloud_series",
        "_sw_announce_channels",
        "_sw_channel_places",
        "_sw_mean_cloud",
        "_sw_night_clouds",
        "_sw_utc_day_dark_hours",
        "_sw_poll_watches",
        "_sw_poll_episode",
        "_run_spaceweather_poll",
        "_deliver_spaceweather_line",
        "_schedule_spaceweather_poll",
        "_enqueue_spaceweather_poll",
    ):
        setattr(obj, name, getattr(LLM, name).__get__(obj))
    return obj


@pytest.fixture
def announcing_plugin(status_plugin):
    """status_plugin plus the announcer's collaborators."""
    from unittest.mock import MagicMock

    from llm import statuspage
    from llm.plugin import LLM

    plugin = status_plugin
    plugin._STATUS_ANNOUNCE_MAX_PER_HOUR = LLM._STATUS_ANNOUNCE_MAX_PER_HOUR
    plugin._STATUS_ANNOUNCE_MAX_LEN = LLM._STATUS_ANNOUNCE_MAX_LEN
    plugin._status_read_cache = {
        "https://status.claude.com": statuspage.Snapshot(
            page_name="Claude",
            page_url="https://status.claude.com",
            indicator="minor",
            description="Partial System Outage",
            components={"Claude API (api.anthropic.com)": "degraded_performance"},
            incidents={},
            fetched_at=plugin._now,
        )
    }
    plugin._status_state = {}
    plugin._registry.update(
        {
            "statusPageUrls": ["https://status.claude.com", "https://www.githubstatus.com"],
            "assistantSystemPrompt": "",
        }
    )
    plugin._status_announce_times = []
    plugin._announce_channels = {"#test": True}
    plugin._all_known_channels = lambda: set(plugin._announce_channels)
    plugin.registryValue = lambda key, channel=None, *a, **k: (
        plugin._announce_channels.get(channel, False)
        if key == "statusAnnounce"
        else plugin._registry.get(key)
    )
    plugin._sent_text = []

    plugin.llm_service = MagicMock()
    plugin.llm_service.sanitize_output = MagicMock(side_effect=lambda t: t or "")
    plugin._collapse_for_irc = MagicMock(side_effect=lambda t: t)

    def fake_privmsg(target, text):
        plugin._sent_text.append(text)
        msg = MagicMock()
        msg.args = (target, text)
        return msg

    plugin._safe_privmsg = MagicMock(side_effect=fake_privmsg)
    plugin._safe_queue = MagicMock(return_value=True)
    plugin._irc_for_channel = MagicMock(return_value=MagicMock())
    plugin._status_rewrite = MagicMock(return_value=None)
    # Must be bound, not left as a MagicMock attribute: _announce_status
    # branches on its return value, and an unbound mock returns a truthy Mock
    # that marks every incident announced while sending nothing.
    plugin._deliver_status_line = LLM._deliver_status_line.__get__(plugin)
    plugin._announce_status = LLM._announce_status.__get__(plugin)
    plugin._status_rewrite_ok = LLM._status_rewrite_ok.__get__(plugin)
    plugin._status_announce_budget_ok = LLM._status_announce_budget_ok.__get__(plugin)
    return plugin
