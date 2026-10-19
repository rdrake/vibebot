"""Tests for @renders — showing, cancelling and clearing the video queue.

The queue these tests inspect is the ``pending_tasks`` table, so they run
against a real temporary database rather than a mocked one: ownership,
state filtering and deletion are all SQL, and a MagicMock db would assert
only that a call happened, not that the right row survived it.

``cancel_video`` is best effort by design. The delivery row goes away
whether or not the box accepts ``DELETE /v1/videos/<id>``, because a row
left behind would post a clip the user already took back, while a job left
running on the box only costs GPU time nobody sees.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

import pytest
from llm.identity import Caller, Identity
from llm.service import LLMService

if TYPE_CHECKING:
    from collections.abc import Callable
    from unittest.mock import Mock

    from llm.persistence import LLMDatabase
    from pytest_mock import MockerFixture

_URL = "http://video.example.com:14205"
# Low-entropy on purpose: a realistic-looking token trips the gitleaks
# pre-commit hook and nothing here reads the value beyond "is it set".
_KEY = "not-a-real-token-for-tests"

_PROMPT = "A lanky, bearded Jesus in a leather jacket rides a moped"


@pytest.fixture
def animate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A configured animate deployment: URL in the registry, key in the env."""
    monkeypatch.setenv("ANIMATE_API_KEY", _KEY)


def _service(make_service: Callable[..., tuple[LLMService, Mock]], **overrides: Any):
    overrides.setdefault("animateApiUrl", _URL)
    return make_service(**overrides)


@pytest.fixture
def renders_env(plugin_env, test_db: LLMDatabase):
    """``plugin_env`` with a real database and a quiesced typing refresher.

    Two departures from the bare fixture, both so assertions mean something:

    * ``plugin.db`` is a real SQLite database — @renders reads and deletes
      rows, and a mock cannot tell a filtered row from a deleted one.
    * the render-typing thread is stopped, so ``_render_typing_wake`` only
      moves when the command under test moves it.

    ``_format_duration`` is bound off the real service for the same reason
    the conftest binds the real ``_status_*`` readers: the mock's return
    value would render as ``<MagicMock ...>`` inside the listing.
    """
    plugin, irc, msg = plugin_env
    mock_db = plugin.db
    plugin.db = test_db
    plugin.llm_service._format_duration = LLMService._format_duration

    plugin._render_typing_stop.set()
    plugin._render_typing_wake.set()
    plugin._render_typing_thread.join(timeout=2.0)
    assert not plugin._render_typing_thread.is_alive()
    plugin._render_typing_stop.clear()
    plugin._render_typing_wake.clear()

    yield plugin, irc, msg

    # Hand the mock back before plugin_env's die() runs: die() sweeps
    # reminders and then closes the database, and test_db is closed by its
    # own teardown, which would make that sweep raise.
    plugin.db = mock_db


def _seed(
    db: LLMDatabase,
    *,
    nick: str = "alice",
    account: str | None = "alice",
    task_type: str = "animate",
    job_id: str = "job-421",
    prompt: str = _PROMPT,
    age: float = 120.0,
    ttl: float = 600.0,
    request_data: str | None = None,
) -> int:
    """Insert one pending task and return its id. ``ttl`` < 0 makes it expired."""
    now = time.time()
    return db.save_pending_task(
        task_type=task_type,
        nick=nick,
        reply_target="#test",
        is_channel=True,
        prompt_preview=prompt,
        model="wan2.2",
        request_data=json.dumps({"job_id": job_id}) if request_data is None else request_data,
        submitted_at=now - age,
        expires_at=now + ttl,
        next_attempt_at=now,
        account=account,
    )


def _reply(irc) -> str:
    """The single line @renders sent."""
    assert irc.reply.call_count == 1, f"expected one reply, got {irc.reply.call_args_list}"
    return irc.reply.call_args.args[0]


def _alice(plugin, mocker: MockerFixture) -> None:
    """Make the caller alice, identified."""
    mocker.patch.object(
        plugin,
        "_resolve_caller",
        return_value=Caller(Identity(raw_nick="alice", account="alice"), "#test"),
    )


def _admin(mocker: MockerFixture, *, yes: bool) -> None:
    """Grant or withhold the admin capability (llm.* stays granted for wrap)."""
    mocker.patch(
        "llm.plugin.ircdb.checkCapability",
        side_effect=lambda _prefix, cap: yes or cap.startswith("llm."),
    )


class TestRendersList:
    def test_empty_queue(self, renders_env) -> None:
        """GIVEN nothing queued WHEN @renders THEN says nothing is rendering."""
        plugin, irc, msg = renders_env

        plugin.renders(irc, msg, [])

        assert "Nothing is rendering" in _reply(irc)

    def test_znc_playback_is_ignored(self, renders_env) -> None:
        """GIVEN a replayed line WHEN @renders THEN nothing is said."""
        plugin, irc, msg = renders_env
        msg.time = plugin.startup_time - 60

        plugin.renders(irc, msg, [])

        irc.reply.assert_not_called()
        irc.error.assert_not_called()

    def test_lists_in_submission_order_with_positions(
        self, renders_env, mocker: MockerFixture
    ) -> None:
        """GIVEN a mixed table WHEN @renders THEN only live animate rows, oldest first."""
        plugin, irc, msg = renders_env
        # Frozen so the age assertion cannot drift on a slow runner. _seed
        # reads the same patched clock, so submitted_at is exactly now - age.
        mocker.patch("llm.plugin.time.time", return_value=time.time())

        first = _seed(plugin.db, nick="alice", account="alice", age=200, prompt="alice clip")
        second = _seed(plugin.db, nick="bob", account="bob", age=100, prompt="bob clip")
        _seed(plugin.db, task_type="draw", prompt="a drawn thing")
        _seed(plugin.db, nick="carol", account="carol", prompt="expired clip", ttl=-10)

        plugin.renders(irc, msg, [])

        line = _reply(irc)
        assert line.index(f"#{first} alice") < line.index(f"#{second} bob")
        assert "1st" in line
        assert "2nd" in line
        assert "a drawn thing" not in line
        assert "expired clip" not in line
        # The age comes from the real duration helper, not a MagicMock repr.
        assert "3m 20s ago" in line

    def test_ready_rows_are_not_listed(self, renders_env) -> None:
        """GIVEN a clip already rendered WHEN @renders THEN it is not in the queue."""
        plugin, irc, msg = renders_env

        task_id = _seed(plugin.db, prompt="already rendered")
        plugin.db.update_task_for_delivery(task_id, "ready", "{}")

        plugin.renders(irc, msg, [])

        assert "Nothing is rendering" in _reply(irc)

    def test_caps_at_six_entries(self, renders_env) -> None:
        """GIVEN eight queued clips WHEN @renders THEN six entries plus a count."""
        plugin, irc, msg = renders_env

        for i in range(8):
            _seed(plugin.db, prompt=f"clip {i}", age=800 - i)

        plugin.renders(irc, msg, [])

        line = _reply(irc)
        assert "clip 5" in line
        assert "clip 6" not in line
        assert "+2 more" in line
        assert line.count(" | ") == 6  # six entries, then the "+2 more" tail
        # Short nicks and short prompts: _RENDERS_LIST_MAX is what capped
        # this listing, not the byte budget.
        assert len(line.encode()) <= plugin._RENDERS_LINE_BUDGET

    def test_long_entries_are_cut_to_a_byte_budget(self, renders_env) -> None:
        """GIVEN six long entries WHEN @renders THEN one IRC line, and "+N more" survives.

        A 30-character nick makes an entry half again as long as a short one.
        Six of those overflow 512 bytes once the PRIVMSG prefix is counted,
        and Limnoria would park the tail -- the "+N more" count and the last
        ids -- behind @more, which is the part people came for.

        380 bytes, tail included: the "+N more" is appended after the loop, so
        the budget has to be checked against the finished line, not the line
        it was measured on.
        """
        plugin, irc, msg = renders_env

        for i in range(6):
            _seed(plugin.db, nick="n" * 30, prompt="p" * 60, age=800 - i)

        plugin.renders(irc, msg, [])

        line = _reply(irc)
        assert len(line.encode()) <= 380
        parts = line.split(" | ")
        shown = len(parts) - 1  # the last part is the "+N more" tail
        assert parts[-1] == f"+{6 - shown} more"
        assert 1 <= shown < 6, "the budget should have truncated, but shown something"

    def test_the_more_tail_is_inside_the_budget_not_added_to_it(self, renders_env) -> None:
        """GIVEN entries that just fit WHEN the tail is appended THEN still one line.

        The sizes here are picked to land on the boundary: five of these
        entries measure 377 bytes, under the 380 the loop checks, and the
        " | +1 more" the loop then appends takes the sent line to 387. Room
        for the tail has to be reserved while the entries are being chosen,
        not discovered after.
        """
        plugin, irc, msg = renders_env

        for i in range(6):
            _seed(plugin.db, nick="n" * 15, prompt="p" * 60, age=800 - i)

        plugin.renders(irc, msg, [])

        line = _reply(irc)
        assert len(line.encode()) <= 380
        assert line.split(" | ")[-1].endswith(" more"), "the tail is what the budget must cover"

    def test_long_prompt_is_cut(self, renders_env) -> None:
        """GIVEN a long prompt WHEN @renders THEN it is elided at 30 characters."""
        plugin, irc, msg = renders_env

        _seed(plugin.db, prompt=_PROMPT)

        plugin.renders(irc, msg, [])

        line = _reply(irc)
        assert f'"{_PROMPT[:30]}…"' in line
        assert "leather" not in line


class TestRendersCancel:
    def test_owner_of_row_can_cancel_and_box_is_asked(
        self, renders_env, mocker: MockerFixture
    ) -> None:
        """GIVEN her own clip WHEN alice cancels THEN the box is told and the row goes."""
        plugin, irc, msg = renders_env
        _alice(plugin, mocker)
        plugin.llm_service.cancel_video.return_value = True

        task_id = _seed(plugin.db, nick="alice", account="alice")

        plugin.renders(irc, msg, ["cancel", str(task_id)])

        plugin.llm_service.cancel_video.assert_called_once_with("job-421")
        assert plugin.db.load_pending_tasks("animate") == []
        assert _reply(irc) == f"Cancelled #{task_id}."
        assert plugin._render_typing_wake.is_set()

    def test_hash_prefixed_id_is_accepted(self, renders_env, mocker: MockerFixture) -> None:
        """GIVEN the id as printed WHEN cancelled with its # THEN it still matches."""
        plugin, irc, msg = renders_env
        _alice(plugin, mocker)
        plugin.llm_service.cancel_video.return_value = True

        task_id = _seed(plugin.db, nick="alice", account="alice")

        plugin.renders(irc, msg, ["cancel", f"#{task_id}"])

        assert plugin.db.load_pending_tasks("animate") == []
        assert _reply(irc) == f"Cancelled #{task_id}."

    def test_box_refusal_is_reported_but_row_still_deleted(
        self, renders_env, mocker: MockerFixture
    ) -> None:
        """GIVEN the box refuses WHEN cancelling THEN the row goes and the reply says so."""
        plugin, irc, msg = renders_env
        _alice(plugin, mocker)
        plugin.llm_service.cancel_video.return_value = False

        task_id = _seed(plugin.db, nick="alice", account="alice")

        plugin.renders(irc, msg, ["cancel", str(task_id)])

        assert plugin.db.load_pending_tasks("animate") == []
        assert _reply(irc) == f"Cancelled #{task_id} (the box kept rendering it)."

    def test_malformed_request_data_still_deletes_the_row(
        self, renders_env, mocker: MockerFixture
    ) -> None:
        """GIVEN no job id on the row WHEN cancelling THEN a plain cancel.

        There is nothing to cancel on the box, so there is no refusal to
        report — saying "the box kept rendering it" would name a call that
        was never made.
        """
        plugin, irc, msg = renders_env
        _alice(plugin, mocker)

        task_id = _seed(plugin.db, nick="alice", account="alice", request_data="not json")

        plugin.renders(irc, msg, ["cancel", str(task_id)])

        plugin.llm_service.cancel_video.assert_not_called()
        assert plugin.db.load_pending_tasks("animate") == []
        assert _reply(irc) == f"Cancelled #{task_id}."

    def test_other_users_row_is_refused_for_non_admin(
        self, renders_env, mocker: MockerFixture
    ) -> None:
        """GIVEN bob's clip WHEN alice cancels it THEN it is refused and kept."""
        plugin, irc, msg = renders_env
        _alice(plugin, mocker)
        _admin(mocker, yes=False)

        task_id = _seed(plugin.db, nick="bob", account="bob")

        plugin.renders(irc, msg, ["cancel", str(task_id)])

        assert len(plugin.db.load_pending_tasks("animate")) == 1
        plugin.llm_service.cancel_video.assert_not_called()
        assert _reply(irc) == f"#{task_id} isn't yours."

    def test_nick_match_owns_the_row_when_unidentified(
        self, renders_env, mocker: MockerFixture
    ) -> None:
        """GIVEN an account-less row WHEN its nick cancels THEN ownership falls back to nick."""
        plugin, irc, msg = renders_env
        mocker.patch.object(
            plugin,
            "_resolve_caller",
            return_value=Caller(Identity(raw_nick="Alice", account=None), "#test"),
        )
        _admin(mocker, yes=False)
        plugin.llm_service.cancel_video.return_value = True

        task_id = _seed(plugin.db, nick="alice", account=None)

        plugin.renders(irc, msg, ["cancel", str(task_id)])

        assert plugin.db.load_pending_tasks("animate") == []
        assert _reply(irc) == f"Cancelled #{task_id}."

    def test_nick_squatter_cannot_cancel_an_identified_users_row(
        self, renders_env, mocker: MockerFixture
    ) -> None:
        """GIVEN a row with an account WHEN an unidentified same-nick caller cancels THEN refused.

        ``Identity.matches`` falls back to raw nick when either side lacks an
        account, so on its own it would hand alice's queued clip to whoever
        grabbed the nick while she was disconnected. A row that carries an
        account needs an identified caller who matches on it.
        """
        plugin, irc, msg = renders_env
        mocker.patch.object(
            plugin,
            "_resolve_caller",
            return_value=Caller(Identity(raw_nick="alice", account=None), "#test"),
        )
        _admin(mocker, yes=False)

        task_id = _seed(plugin.db, nick="alice", account="alice")

        plugin.renders(irc, msg, ["cancel", str(task_id)])

        assert len(plugin.db.load_pending_tasks("animate")) == 1
        plugin.llm_service.cancel_video.assert_not_called()
        assert _reply(irc) == f"#{task_id} isn't yours."

    def test_admin_can_cancel_anyone(self, renders_env, mocker: MockerFixture) -> None:
        """GIVEN bob's clip WHEN an admin cancels it THEN it goes."""
        plugin, irc, msg = renders_env
        _alice(plugin, mocker)
        _admin(mocker, yes=True)
        plugin.llm_service.cancel_video.return_value = True

        task_id = _seed(plugin.db, nick="bob", account="bob")

        plugin.renders(irc, msg, ["cancel", str(task_id)])

        assert plugin.db.load_pending_tasks("animate") == []
        assert _reply(irc) == f"Cancelled #{task_id}."

    def test_unknown_id(self, renders_env) -> None:
        """GIVEN no such clip WHEN cancelling THEN says there is no pending clip."""
        plugin, irc, msg = renders_env

        plugin.renders(irc, msg, ["cancel", "999"])

        assert _reply(irc) == "No pending clip #999."

    def test_expired_row_is_not_cancellable(self, renders_env, mocker: MockerFixture) -> None:
        """GIVEN an expired clip WHEN cancelling THEN it is invisible, like the listing."""
        plugin, irc, msg = renders_env
        _alice(plugin, mocker)

        task_id = _seed(plugin.db, nick="alice", account="alice", ttl=-10)

        plugin.renders(irc, msg, ["cancel", str(task_id)])

        assert len(plugin.db.load_pending_tasks("animate")) == 1
        assert _reply(irc) == f"No pending clip #{task_id}."

    @pytest.mark.parametrize("bad_id", ["abc", "\u00b2", "12abc", ""])
    def test_non_decimal_id_gets_the_usage_line(self, renders_env, bad_id: str) -> None:
        """GIVEN a non-decimal id WHEN cancelling THEN the usage line, not a traceback.

        "\u00b2".isdigit() is True and int("\u00b2") raises, so isdigit here would
        take a ValueError out of the command.
        """
        plugin, irc, msg = renders_env

        plugin.renders(irc, msg, ["cancel", bad_id] if bad_id else ["cancel"])

        irc.reply.assert_not_called()
        assert "Usage: @renders" in irc.error.call_args.args[0]

    def test_bad_usage_errors(self, renders_env) -> None:
        """GIVEN nonsense WHEN @renders THEN the usage line."""
        plugin, irc, msg = renders_env

        plugin.renders(irc, msg, ["explode"])

        irc.reply.assert_not_called()
        irc.error.assert_called_once()
        assert "Usage: @renders" in irc.error.call_args.args[0]


class TestRendersCapability:
    def test_caller_without_llm_animate_is_refused(
        self, renders_env, mocker: MockerFixture
    ) -> None:
        """GIVEN no llm.animate WHEN @renders cancel THEN the wrap gate stops it.

        Locks the gate in: drop the checkCapability converter and this goes
        red, because @renders reaches persisted, account-owned rows.
        """
        plugin, irc, msg = renders_env
        _alice(plugin, mocker)
        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=False)

        task_id = _seed(plugin.db, nick="alice", account="alice")

        plugin.renders(irc, msg, ["cancel", str(task_id)])

        assert len(plugin.db.load_pending_tasks("animate")) == 1
        plugin.llm_service.cancel_video.assert_not_called()
        irc.reply.assert_not_called()
        irc.errorNoCapability.assert_called_once_with("llm.animate", Raise=True)


class TestRendersClear:
    def test_non_admin_refused(self, renders_env, mocker: MockerFixture) -> None:
        """GIVEN no admin capability WHEN clearing THEN refused and nothing deleted."""
        plugin, irc, msg = renders_env
        _admin(mocker, yes=False)
        _seed(plugin.db)

        plugin.renders(irc, msg, ["clear"])

        assert len(plugin.db.load_pending_tasks("animate")) == 1
        assert "'admin' capability" in _reply(irc)

    def test_admin_clears_all_pending_animate_rows_only(
        self, renders_env, mocker: MockerFixture
    ) -> None:
        """GIVEN a mixed table WHEN an admin clears THEN only live animate rows go."""
        plugin, irc, msg = renders_env
        _admin(mocker, yes=True)
        plugin.llm_service.cancel_video.return_value = True

        for i in range(3):
            _seed(plugin.db, nick=f"user{i}", account=f"user{i}", job_id=f"job-{i}")
        _seed(plugin.db, task_type="draw", prompt="a drawn thing")
        ready = _seed(plugin.db, prompt="already rendered")
        plugin.db.update_task_for_delivery(ready, "ready", "{}")

        plugin.renders(irc, msg, ["clear"])

        assert plugin.llm_service.cancel_video.call_count == 3
        assert [r.prompt_preview for r in plugin.db.load_pending_tasks("animate")] == [
            "already rendered"
        ]
        assert len(plugin.db.load_pending_tasks("draw")) == 1
        assert _reply(irc) == "Cleared 3 clip(s)."
        assert plugin._render_typing_wake.is_set()


class TestCancelVideo:
    def test_delete_success_codes(self, make_service, animate_env, mocker: MockerFixture) -> None:
        """GIVEN the box accepts WHEN cancelling THEN True and a DELETE was sent."""
        service, _plugin = _service(make_service)

        for code in (200, 202, 204):
            request = mocker.patch.object(service, "_animate_request", return_value=(code, {}))

            assert service.cancel_video("job-1") is True
            request.assert_called_once_with("/v1/videos/job-1", method="DELETE", timeout=5.0)

    def test_cancel_uses_a_short_timeout_not_animate_timeout(
        self, make_service, animate_env, mocker: MockerFixture
    ) -> None:
        """GIVEN a wedged box WHEN cancelling THEN 5s, not the 60s render timeout.

        ``@renders clear`` cancels every queued row in turn, so inheriting
        ``animateTimeout`` would hold the command for a minute per row while
        the user watches nothing happen.
        """
        service, _plugin = _service(make_service, animateTimeout=60)
        request = mocker.patch.object(service, "_animate_request", return_value=(204, {}))

        service.cancel_video("job-1")

        request.assert_called_once_with("/v1/videos/job-1", method="DELETE", timeout=5.0)

    def test_delete_failure_logs_and_returns_false(
        self, make_service, animate_env, mocker: MockerFixture
    ) -> None:
        """GIVEN the box refuses WHEN cancelling THEN False, not an exception."""
        service, _plugin = _service(make_service)
        mocker.patch.object(
            service,
            "_animate_request",
            return_value=(404, {"error": {"message": "gone"}}),
        )
        log = mocker.patch.object(service, "log")

        assert service.cancel_video("job-1") is False
        assert "gone" in str(log.warning.call_args)

    def test_transport_failure_never_raises(
        self, make_service, animate_env, mocker: MockerFixture
    ) -> None:
        """GIVEN the box is unreachable WHEN cancelling THEN False, not an exception."""
        service, _plugin = _service(make_service)
        mocker.patch.object(service, "_animate_request", side_effect=OSError("no route"))
        log = mocker.patch.object(service, "log")

        assert service.cancel_video("job-1") is False
        assert "no route" in str(log.warning.call_args)
