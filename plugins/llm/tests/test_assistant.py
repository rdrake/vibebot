"""Tests for the meta command tool definitions and executor."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from llm import provider
from llm.assistant import (
    _BOOKKEEPING_TOOLS,
    ASSISTANT_TOOL_SPECS,
    ASSISTANT_TOOLS,
    PENDING_TASK_TOOLS,
    AssistantToolExecutor,
    ToolCallbackResult,
    ToolResult,
    get_tools_for_profile,
)
from llm.identity import Caller, Identity
from llm.plugin import LLM, PreflightResult
from llm.profile import PROFILE_CHAT, PROFILE_REMIND_ACTION
from llm.prompts import CHAT_SYSTEM_PROMPT, PENDING_TASKS_GUIDANCE
from llm.service import (
    _REPEAT_RETRY_NUDGE,
    LLMService,
    _depoison_verse_history,
    _is_degraded_reply,
    _is_echo_reply,
    _is_verse_denial,
    _normalize_for_echo,
    _replies_repetitive,
    _strip_degraded,
    _strip_repeated_replies,
    _strip_verse_denials,
    _trim_history_window,
)

from .conftest import (
    make_completion_response,
    make_registry_side_effect,
    make_reminder_row,
    make_tool_call,
    plugin_init_patches,
)

if TYPE_CHECKING:
    from unittest.mock import MagicMock

    from pytest_mock import MockerFixture


def make_executor(*args, **kwargs) -> AssistantToolExecutor:
    """Build an executor for handler-behaviour tests.

    Defaults to ``remind_action`` -- the profile that still advertises the
    whole tool surface. Chat deliberately hides the bookkeeping tools
    (``_BOOKKEEPING_TOOLS``), and these cases are about what a handler does
    once invoked, not about the visibility gate, which has its own tests.
    Callers that ARE testing visibility pass ``route_profile`` explicitly and
    keep whatever they pass.
    """
    kwargs.setdefault("route_profile", PROFILE_REMIND_ACTION)
    # Personal-data tools refuse an accountless caller; that gate has its own
    # tests (TestPersonalDataToolsNeedAccount).
    kwargs.setdefault("account", "testuser")
    return AssistantToolExecutor(*args, **kwargs)


class TestMetaTools:
    """GIVEN the ASSISTANT_TOOLS list WHEN inspected THEN it has correct structure."""

    def test_tool_count(self) -> None:
        """GIVEN ASSISTANT_TOOLS WHEN counted THEN has expected number of tools."""
        assert len(ASSISTANT_TOOLS) == 22

    def test_tools_have_function_format(self) -> None:
        """GIVEN each tool WHEN checked THEN follows OpenAI function calling schema."""
        for tool in ASSISTANT_TOOLS:
            assert tool["type"] == "function"
            assert "function" in tool
            fn = tool["function"]
            assert "name" in fn
            assert "description" in fn
            assert "parameters" in fn


class TestAssistantToolExecutor:
    """Tests for AssistantToolExecutor dispatching tool calls."""

    @pytest.fixture
    def mock_db(self, mocker: MockerFixture) -> MagicMock:
        db = mocker.MagicMock()
        db.get_instruction.return_value = "respond in haiku"
        db.get_memories.return_value = [
            mocker.MagicMock(id=1, fact="likes Python"),
            mocker.MagicMock(id=2, fact="owns a cat"),
        ]
        db.save_instruction.return_value = None
        db.delete_instruction.return_value = True
        db.save_memory.return_value = 3
        db.delete_memory.return_value = True
        db.update_memory.return_value = True
        db.delete_all_memories.return_value = 2
        return db

    @pytest.fixture
    def mock_context(self, mocker: MockerFixture) -> MagicMock:
        ctx = mocker.MagicMock()
        ctx.clear.return_value = True
        return ctx

    @pytest.fixture
    def mock_cleanup_fn(self, mocker: MockerFixture) -> MagicMock:
        fn = mocker.MagicMock()
        fn.return_value = ToolCallbackResult(
            True, "Before: 8 | dropped: 2, merged: 4 \u2192 2 | after: 4"
        )
        return fn

    @pytest.fixture
    def mock_list_pending_tasks_fn(self, mocker: MockerFixture) -> MagicMock:
        fn = mocker.MagicMock()
        fn.return_value = [
            {
                "kind": "reminder",
                "id": "abc123",
                "channel": "#test",
                "description": "check build",
            },
            {
                "kind": "reminder",
                "id": "def456",
                "channel": "#test",
                "description": "deploy app",
            },
        ]
        return fn

    @pytest.fixture
    def mock_set_reminder_fn(self, mocker: MockerFixture) -> MagicMock:
        fn = mocker.MagicMock()
        fn.return_value = ToolCallbackResult(True, "I'll remind you in 1 hour")
        return fn

    @pytest.fixture
    def mock_cancel_pending_task_fn(self, mocker: MockerFixture) -> MagicMock:
        fn = mocker.MagicMock()
        fn.return_value = {
            "status": "ok",
            "kind": "reminder",
            "id": "abc123",
            "message": "Deleted reminder abc123.",
        }
        return fn

    @pytest.fixture
    def mock_cancel_all_pending_tasks_fn(self, mocker: MockerFixture) -> MagicMock:
        fn = mocker.MagicMock()
        fn.return_value = {
            "reminders_message": "Cancelled 2 reminders.",
            "scheduled_tasks_cancelled": 0,
        }
        return fn

    @pytest.fixture
    def executor(
        self,
        mock_db: MagicMock,
        mock_context: MagicMock,
        mock_cleanup_fn: MagicMock,
        mock_list_pending_tasks_fn: MagicMock,
        mock_set_reminder_fn: MagicMock,
        mock_cancel_pending_task_fn: MagicMock,
        mock_cancel_all_pending_tasks_fn: MagicMock,
    ) -> AssistantToolExecutor:
        return make_executor(
            db=mock_db,
            context=mock_context,
            nick="testuser",
            channel="#test",
            cleanup_fn=mock_cleanup_fn,
            list_pending_tasks_fn=mock_list_pending_tasks_fn,
            set_reminder_fn=mock_set_reminder_fn,
            cancel_pending_task_fn=mock_cancel_pending_task_fn,
            cancel_all_pending_tasks_fn=mock_cancel_all_pending_tasks_fn,
            # These cases exercise handler behaviour, not the visibility gate
            # (which has its own test). remind_action is the profile that still
            # advertises the whole surface; chat deliberately no longer does.
            route_profile=PROFILE_REMIND_ACTION,
        )

    def test_get_instruction_tool_is_gone(self, executor: AssistantToolExecutor) -> None:
        """The read tool was removed — the instruction is injected as data."""
        result = executor.execute("get_instruction", {})
        assert "Unknown tool" in result.content

    def test_set_instruction(self, executor: AssistantToolExecutor, mock_db: MagicMock) -> None:
        """GIVEN set_instruction tool WHEN called THEN saves instruction."""
        result = executor.execute("set_instruction", {"text": "be brief"})
        mock_db.save_instruction.assert_called_once_with("testuser", "be brief")
        assert "ok" in result.content.lower()

    def test_clear_instruction(self, executor: AssistantToolExecutor, mock_db: MagicMock) -> None:
        """GIVEN clear_instruction tool WHEN called THEN deletes instruction."""
        result = executor.execute("clear_instruction", {})
        mock_db.delete_instruction.assert_called_once_with("testuser")
        assert "clear" in result.content.lower()

    def test_list_memories(self, executor: AssistantToolExecutor) -> None:
        """GIVEN list_memories tool WHEN called THEN returns formatted memories."""
        result = executor.execute("list_memories", {})
        assert "likes Python" in result.content
        assert "owns a cat" in result.content

    def test_save_memory(self, executor: AssistantToolExecutor, mock_db: MagicMock) -> None:
        """GIVEN save_memory tool WHEN called THEN saves to db."""
        result = executor.execute("save_memory", {"text": "prefers vim"})
        mock_db.save_memory.assert_called_once_with("testuser", "prefers vim", "#test")
        assert "saved" in result.content.lower() or "3" in result.content

    def test_delete_memory(self, executor: AssistantToolExecutor, mock_db: MagicMock) -> None:
        """GIVEN delete_memory tool WHEN called THEN deletes by ID."""
        result = executor.execute("delete_memory", {"id": 1})
        mock_db.delete_memory.assert_called_once_with("testuser", 1)
        assert "delete" in result.content.lower()

    def test_delete_memory_not_found(
        self, executor: AssistantToolExecutor, mock_db: MagicMock
    ) -> None:
        """GIVEN delete_memory tool WHEN ID not found THEN returns error."""
        mock_db.delete_memory.return_value = False
        result = executor.execute("delete_memory", {"id": 999})
        assert "not found" in result.content.lower() or "error" in result.content.lower()

    def test_update_memory(self, executor: AssistantToolExecutor, mock_db: MagicMock) -> None:
        """GIVEN update_memory tool WHEN called THEN updates in db."""
        result = executor.execute("update_memory", {"id": 1, "text": "loves Python"})
        mock_db.update_memory.assert_called_once_with("testuser", 1, "loves Python")
        assert "update" in result.content.lower()

    def test_clear_memories(self, executor: AssistantToolExecutor, mock_db: MagicMock) -> None:
        """GIVEN clear_memories tool WHEN called THEN deletes all."""
        result = executor.execute("clear_memories", {})
        mock_db.delete_all_memories.assert_called_once_with("testuser")
        assert "2" in result.content  # count returned

    def test_forget_context(self, executor: AssistantToolExecutor, mock_context: MagicMock) -> None:
        """GIVEN forget_context tool WHEN called THEN clears context for channel."""
        result = executor.execute("forget_context", {})
        mock_context.clear.assert_called_once_with("testuser", "#test")
        assert "clear" in result.content.lower()

    def test_unknown_tool(self, executor: AssistantToolExecutor) -> None:
        """GIVEN unknown tool name WHEN called THEN returns error."""
        result = executor.execute("launch_missiles", {})
        assert "error" in result.content.lower() or "unknown" in result.content.lower()

    def test_executor_catches_exceptions(
        self, executor: AssistantToolExecutor, mock_db: MagicMock
    ) -> None:
        """GIVEN tool raises exception WHEN executed THEN returns error string."""
        mock_db.get_memories.side_effect = RuntimeError("db error")
        result = executor.execute("list_memories", {})
        assert "error" in result.content.lower()

    def test_execute_denies_when_capability_missing(
        self,
        mock_db: MagicMock,
        mock_context: MagicMock,
        mock_cleanup_fn: MagicMock,
        mock_list_pending_tasks_fn: MagicMock,
        mock_set_reminder_fn: MagicMock,
        mock_cancel_pending_task_fn: MagicMock,
    ) -> None:
        """GIVEN missing tool capability WHEN executed THEN dispatch denies it server-side."""
        executor = make_executor(
            db=mock_db,
            context=mock_context,
            nick="testuser",
            channel="#test",
            capabilities=frozenset(),
            cleanup_fn=mock_cleanup_fn,
            list_pending_tasks_fn=mock_list_pending_tasks_fn,
            set_reminder_fn=mock_set_reminder_fn,
            cancel_pending_task_fn=mock_cancel_pending_task_fn,
        )

        result = executor.execute("list_memories", {})

        assert "not allowed" in result.content.lower() or "capability" in result.content.lower()
        mock_db.get_memories.assert_not_called()

    def test_execute_denies_when_route_profile_not_visible(
        self,
        mock_db: MagicMock,
        mock_context: MagicMock,
        mock_cleanup_fn: MagicMock,
        mock_list_pending_tasks_fn: MagicMock,
        mock_set_reminder_fn: MagicMock,
        mock_cancel_pending_task_fn: MagicMock,
    ) -> None:
        """GIVEN a hidden route profile WHEN executed THEN dispatch denies it server-side."""
        executor = make_executor(
            db=mock_db,
            context=mock_context,
            nick="testuser",
            channel="#test",
            route_profile="draw",
            cleanup_fn=mock_cleanup_fn,
            list_pending_tasks_fn=mock_list_pending_tasks_fn,
            set_reminder_fn=mock_set_reminder_fn,
            cancel_pending_task_fn=mock_cancel_pending_task_fn,
        )

        result = executor.execute("list_memories", {})

        assert "not allowed" in result.content.lower() or "profile" in result.content.lower()
        mock_db.get_memories.assert_not_called()

    def test_get_usage(self, executor: AssistantToolExecutor, mock_db: MagicMock) -> None:
        """GIVEN get_usage tool WHEN called THEN returns user's usage summary."""
        from llm.persistence import UsageSummary

        mock_db.get_usage_summary_for_nick.return_value = UsageSummary(
            total_requests=47,
            total_prompt_tokens=12000,
            total_completion_tokens=3000,
            total_cost=0.12,
        )
        result = executor.execute("get_usage", {})
        assert "47" in result.content
        assert "0.12" in result.content
        mock_db.get_usage_summary_for_nick.assert_called_once()

    def test_get_channel_usage(self, executor: AssistantToolExecutor, mock_db: MagicMock) -> None:
        """GIVEN get_channel_usage tool WHEN called THEN returns channel summary."""
        from llm.persistence import UsageSummary

        mock_db.get_usage_summary_for_channel.return_value = UsageSummary(
            total_requests=200,
            total_prompt_tokens=50000,
            total_completion_tokens=10000,
            total_cost=0.85,
        )
        result = executor.execute("get_channel_usage", {})
        assert "200" in result.content
        assert "0.85" in result.content
        mock_db.get_usage_summary_for_channel.assert_called_once()

    def test_cleanup_memories(
        self, executor: AssistantToolExecutor, mock_cleanup_fn: MagicMock
    ) -> None:
        """GIVEN cleanup_memories tool WHEN called THEN runs cleanup callable."""
        result = executor.execute("cleanup_memories", {})
        mock_cleanup_fn.assert_called_once_with("testuser")
        assert "Before: 8" in result.content

    def test_cleanup_memories_not_available(
        self, mock_db: MagicMock, mock_context: MagicMock
    ) -> None:
        """GIVEN no cleanup_fn WHEN cleanup_memories called THEN returns error."""
        executor = make_executor(db=mock_db, context=mock_context, nick="testuser", channel="#test")
        result = executor.execute("cleanup_memories", {})
        assert "not available" in result.content.lower() or "error" in result.content.lower()

    def test_list_memories_other_user_as_owner(
        self, mock_db: MagicMock, mock_context: MagicMock, mock_cleanup_fn: MagicMock
    ) -> None:
        """GIVEN owner WHEN listing another user's memories THEN allowed."""
        executor = make_executor(
            db=mock_db,
            context=mock_context,
            nick="owner",
            channel="#test",
            is_owner=True,
            cleanup_fn=mock_cleanup_fn,
        )
        result = executor.execute("list_memories", {"nick": "someone"})
        mock_db.get_memories.assert_called_with("someone")
        assert "someone" in result.content

    def test_list_memories_other_user_denied(
        self, mock_db: MagicMock, mock_context: MagicMock
    ) -> None:
        """GIVEN non-owner WHEN listing another user's memories THEN denied."""
        executor = make_executor(
            db=mock_db,
            context=mock_context,
            nick="regular",
            channel="#test",
        )
        result = executor.execute("list_memories", {"nick": "someone"})
        assert "owner" in result.content.lower()
        mock_db.get_memories.assert_not_called()

    def test_delete_memory_other_user_as_owner(
        self, mock_db: MagicMock, mock_context: MagicMock
    ) -> None:
        """GIVEN owner WHEN deleting another user's memory THEN allowed."""
        executor = make_executor(
            db=mock_db,
            context=mock_context,
            nick="owner",
            channel="#test",
            is_owner=True,
        )
        executor.execute("delete_memory", {"id": 5, "nick": "someone"})
        mock_db.delete_memory.assert_called_once_with("someone", 5)

    def test_clear_memories_other_user_denied(
        self, mock_db: MagicMock, mock_context: MagicMock
    ) -> None:
        """GIVEN non-owner WHEN clearing another user's memories THEN denied."""
        executor = make_executor(
            db=mock_db,
            context=mock_context,
            nick="regular",
            channel="#test",
        )
        result = executor.execute("clear_memories", {"nick": "someone"})
        assert "owner" in result.content.lower()
        mock_db.delete_all_memories.assert_not_called()

    def test_cleanup_other_user_as_owner(
        self, mock_db: MagicMock, mock_context: MagicMock, mock_cleanup_fn: MagicMock
    ) -> None:
        """GIVEN owner WHEN cleaning up another user's memories THEN allowed."""
        executor = make_executor(
            db=mock_db,
            context=mock_context,
            nick="owner",
            channel="#test",
            is_owner=True,
            cleanup_fn=mock_cleanup_fn,
        )
        executor.execute("cleanup_memories", {"nick": "someone"})
        mock_cleanup_fn.assert_called_once_with("someone")

    def test_list_pending_tasks(
        self, executor: AssistantToolExecutor, mock_list_pending_tasks_fn: MagicMock
    ) -> None:
        """GIVEN list_pending_tasks tool WHEN called THEN returns merged list."""
        result = executor.execute("list_pending_tasks", {})
        mock_list_pending_tasks_fn.assert_called_once()
        assert "check build" in result.content
        assert "deploy app" in result.content
        assert "abc123" in result.content

    def test_list_pending_tasks_empty(
        self, executor: AssistantToolExecutor, mock_list_pending_tasks_fn: MagicMock
    ) -> None:
        """GIVEN no tasks WHEN list_pending_tasks THEN returns empty message."""
        mock_list_pending_tasks_fn.return_value = []
        result = executor.execute("list_pending_tasks", {})
        assert "no" in result.content.lower() or "[]" in result.content

    def test_set_reminder(
        self, executor: AssistantToolExecutor, mock_set_reminder_fn: MagicMock
    ) -> None:
        """GIVEN set_reminder tool WHEN called THEN schedules via callable."""
        result = executor.execute("set_reminder", {"text": "check build in 1 hour"})
        mock_set_reminder_fn.assert_called_once_with("check build in 1 hour")
        assert "remind" in result.content.lower() or "hour" in result.content.lower()

    def test_cancel_pending_task(
        self, executor: AssistantToolExecutor, mock_cancel_pending_task_fn: MagicMock
    ) -> None:
        """GIVEN cancel_pending_task tool WHEN called THEN dispatches via callable."""
        result = executor.execute("cancel_pending_task", {"id": "abc123"})
        mock_cancel_pending_task_fn.assert_called_once_with("abc123")
        assert "delete" in result.content.lower()

    def test_cancel_pending_task_not_found(
        self,
        executor: AssistantToolExecutor,
        mock_cancel_pending_task_fn: MagicMock,
    ) -> None:
        """GIVEN nonexistent task WHEN cancel_pending_task THEN returns error."""
        mock_cancel_pending_task_fn.return_value = {
            "status": "error",
            "kind": "reminder",
            "id": "xyz",
            "message": "Reminder xyz not found.",
        }
        result = executor.execute("cancel_pending_task", {"id": "xyz"})
        assert "not found" in result.content.lower()

    def test_cancel_all_pending_tasks(
        self,
        executor: AssistantToolExecutor,
        mock_cancel_all_pending_tasks_fn: MagicMock,
    ) -> None:
        """GIVEN cancel_all_pending_tasks tool WHEN called THEN dispatches to callable."""
        result = executor.execute("cancel_all_pending_tasks", {})
        mock_cancel_all_pending_tasks_fn.assert_called_once_with()
        assert "2" in result.content
        assert "ok" in result.content.lower()

    def test_cancel_all_pending_tasks_unavailable(
        self,
        mock_db: MagicMock,
        mock_context: MagicMock,
    ) -> None:
        """GIVEN no callable WHEN cancel_all_pending_tasks THEN returns error."""
        executor = make_executor(
            db=mock_db,
            context=mock_context,
            nick="testuser",
            channel="#test",
        )
        result = executor.execute("cancel_all_pending_tasks", {})
        assert "not available" in result.content.lower()

    # -- Task 6: Structured returns and new callables ----------------------

    def test_executor_accepts_search_fn(
        self, mock_db: MagicMock, mock_context: MagicMock, mocker: MockerFixture
    ) -> None:
        """AssistantToolExecutor accepts search_fn callable."""
        search_fn = mocker.MagicMock(return_value=ToolResult(content="results"))
        executor = make_executor(
            db=mock_db,
            context=mock_context,
            nick="testuser",
            channel="#test",
            search_fn=search_fn,
        )
        assert executor._search_fn is search_fn

    def test_executor_accepts_fetch_fn(
        self, mock_db: MagicMock, mock_context: MagicMock, mocker: MockerFixture
    ) -> None:
        """AssistantToolExecutor accepts fetch_fn callable."""
        fetch_fn = mocker.MagicMock(return_value=ToolResult(content="page"))
        executor = make_executor(
            db=mock_db,
            context=mock_context,
            nick="testuser",
            channel="#test",
            fetch_fn=fetch_fn,
        )
        assert executor._fetch_fn is fetch_fn

    def test_executor_accepts_code_fn(
        self, mock_db: MagicMock, mock_context: MagicMock, mocker: MockerFixture
    ) -> None:
        """AssistantToolExecutor accepts code_fn callable."""
        code_fn = mocker.MagicMock(return_value=ToolResult(content="code"))
        executor = make_executor(
            db=mock_db,
            context=mock_context,
            nick="testuser",
            channel="#test",
            code_fn=code_fn,
        )
        assert executor._code_fn is code_fn

    def test_execute_returns_tool_result(self, executor: AssistantToolExecutor) -> None:
        """execute() returns ToolResult for existing tools."""
        result = executor.execute("set_instruction", {"text": "be brief"})
        assert isinstance(result, ToolResult)
        assert "be brief" in result.content

    def test_execute_denied_returns_tool_result(
        self, mock_db: MagicMock, mock_context: MagicMock
    ) -> None:
        """execute() returns ToolResult for denied tools."""
        executor = make_executor(
            db=mock_db,
            context=mock_context,
            nick="testuser",
            channel="#test",
            capabilities=frozenset(),
        )
        result = executor.execute("list_memories", {})
        assert isinstance(result, ToolResult)
        assert "capability" in result.content.lower() or "not allowed" in result.content.lower()

    def test_executor_tracks_grounding_used(
        self, mock_db: MagicMock, mock_context: MagicMock, mocker: MockerFixture
    ) -> None:
        """Executor sets grounding_used when tool returns grounding_used=True."""
        search_fn = mocker.MagicMock(
            return_value=ToolResult(content="web results", grounding_used=True),
        )
        executor = make_executor(
            db=mock_db,
            context=mock_context,
            nick="testuser",
            channel="#test",
            route_profile="chat",
            search_fn=search_fn,
        )
        assert executor.grounding_used is False
        executor.execute("search_web", {"query": "test"})
        assert executor.grounding_used is True

    def test_executor_accumulates_cost(
        self, mock_db: MagicMock, mock_context: MagicMock, mocker: MockerFixture
    ) -> None:
        """Executor accumulates cost from ToolResult."""
        search_fn = mocker.MagicMock(
            return_value=ToolResult(
                content="results",
                prompt_tokens=100,
                completion_tokens=50,
                cost=0.01,
            ),
        )
        executor = make_executor(
            db=mock_db,
            context=mock_context,
            nick="testuser",
            channel="#test",
            route_profile="chat",
            search_fn=search_fn,
        )
        executor.execute("search_web", {"query": "a"})
        executor.execute("search_web", {"query": "b"})
        assert executor.accumulated_prompt_tokens == 200
        assert executor.accumulated_completion_tokens == 100
        assert executor.accumulated_cost == pytest.approx(0.02)

    # -- Task 7: Tool handlers --------------------------------------------

    def test_tool_search_web_calls_fn(
        self, mock_db: MagicMock, mock_context: MagicMock, mocker: MockerFixture
    ) -> None:
        """search_web handler calls search_fn with query."""
        search_fn = mocker.MagicMock(
            return_value=ToolResult(content="web results"),
        )
        executor = make_executor(
            db=mock_db,
            context=mock_context,
            nick="testuser",
            channel="#test",
            route_profile="chat",
            search_fn=search_fn,
        )
        result = executor.execute("search_web", {"query": "python async"})
        search_fn.assert_called_once_with("python async")
        assert result.content == "web results"

    def test_tool_search_web_capped_per_request(
        self, mock_db: MagicMock, mock_context: MagicMock, mocker: MockerFixture
    ) -> None:
        """The fourth search_web in one request is refused without searching."""
        from llm.assistant import MAX_SEARCHES_PER_REQUEST

        search_fn = mocker.MagicMock(return_value=ToolResult(content="web results"))
        executor = make_executor(
            db=mock_db,
            context=mock_context,
            nick="testuser",
            channel="#test",
            route_profile="chat",
            search_fn=search_fn,
        )
        results = [
            executor.execute("search_web", {"query": f"q{i}"})
            for i in range(MAX_SEARCHES_PER_REQUEST + 2)
        ]
        assert search_fn.call_count == MAX_SEARCHES_PER_REQUEST == 3
        assert all(r.content == "web results" for r in results[:3])
        assert all("Search limit reached" in r.content for r in results[3:])

    def test_search_cap_is_per_executor(
        self, mock_db: MagicMock, mock_context: MagicMock, mocker: MockerFixture
    ) -> None:
        """A new request (new executor) gets a fresh search budget."""
        search_fn = mocker.MagicMock(return_value=ToolResult(content="web results"))

        def run() -> None:
            executor = make_executor(
                db=mock_db,
                context=mock_context,
                nick="testuser",
                channel="#test",
                route_profile="chat",
                search_fn=search_fn,
            )
            for i in range(3):
                executor.execute("search_web", {"query": f"q{i}"})

        run()
        run()
        assert search_fn.call_count == 6

    def test_tool_search_web_no_fn_returns_error(
        self, mock_db: MagicMock, mock_context: MagicMock
    ) -> None:
        """search_web returns error when search_fn is None."""
        executor = make_executor(
            db=mock_db,
            context=mock_context,
            nick="testuser",
            channel="#test",
            route_profile="chat",
        )
        result = executor.execute("search_web", {"query": "test"})
        assert "unavailable" in result.content.lower() or "error" in result.content.lower()

    def test_tool_fetch_url_calls_fn(
        self, mock_db: MagicMock, mock_context: MagicMock, mocker: MockerFixture
    ) -> None:
        """fetch_url handler calls fetch_fn with url."""
        fetch_fn = mocker.MagicMock(
            return_value=ToolResult(content="page content"),
        )
        executor = make_executor(
            db=mock_db,
            context=mock_context,
            nick="testuser",
            channel="#test",
            route_profile="chat",
            fetch_fn=fetch_fn,
        )
        result = executor.execute("fetch_url", {"url": "https://example.com"})
        fetch_fn.assert_called_once_with("https://example.com")
        assert result.content == "page content"

    def test_tool_fetch_url_no_fn_returns_error(
        self, mock_db: MagicMock, mock_context: MagicMock
    ) -> None:
        """fetch_url returns error when fetch_fn is None."""
        executor = make_executor(
            db=mock_db,
            context=mock_context,
            nick="testuser",
            channel="#test",
            route_profile="chat",
        )
        result = executor.execute("fetch_url", {"url": "https://example.com"})
        assert "unavailable" in result.content.lower() or "error" in result.content.lower()

    def test_tool_generate_code_calls_fn(
        self, mock_db: MagicMock, mock_context: MagicMock, mocker: MockerFixture
    ) -> None:
        """generate_code handler calls code_fn with prompt."""
        code_fn = mocker.MagicMock(
            return_value=ToolResult(content="https://paste.example/abc"),
        )
        executor = make_executor(
            db=mock_db,
            context=mock_context,
            nick="testuser",
            channel="#test",
            route_profile="code",
            capabilities=frozenset({"llm.ask", "llm.code"}),
            code_fn=code_fn,
        )
        result = executor.execute("generate_code", {"prompt": "fizzbuzz in rust"})
        code_fn.assert_called_once_with("fizzbuzz in rust")
        assert result.content == "https://paste.example/abc"

    def test_tool_generate_code_no_fn_returns_error(
        self, mock_db: MagicMock, mock_context: MagicMock
    ) -> None:
        """generate_code returns error when code_fn is None."""
        executor = make_executor(
            db=mock_db,
            context=mock_context,
            nick="testuser",
            channel="#test",
            route_profile="code",
            capabilities=frozenset({"llm.ask", "llm.code"}),
        )
        result = executor.execute("generate_code", {"prompt": "hello world"})
        assert "unavailable" in result.content.lower() or "error" in result.content.lower()


# =========================================================================
# assistant_completion() service-level tests
# =========================================================================


class TestMetaCompletion:
    """Tests for LLMService.assistant_completion() tool-calling loop."""

    @pytest.fixture
    def service(self, make_service) -> LLMService:  # type: ignore[no-untyped-def]
        svc, _plugin = make_service(assistantModel="gpt-4")
        return svc

    def test_text_response_no_tools(self, service: LLMService, mocker: MockerFixture) -> None:
        """GIVEN LLM returns text WHEN no tool calls THEN returns text."""
        mock_response = make_completion_response("Done \u2014 instruction set.")

        mocker.patch(
            "llm.service.provider.chat",
            return_value=mock_response,
        )

        result = service.assistant_completion(
            prompt="set my instruction to haiku",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
        )

        assert result.content == "Done \u2014 instruction set."
        assert result.error is None

    def test_tool_call_then_text(self, service: LLMService, mocker: MockerFixture) -> None:
        """GIVEN LLM calls a tool then responds WHEN executed THEN tool runs."""
        # Canary for the shared conftest builders (consolidation #1): identical
        # shape to the hand-rolled MagicMock scaffolding, one source of truth.
        tool_call = make_tool_call(
            "set_instruction", {"text": "respond in haiku"}, call_id="call_1"
        )
        first_response = make_completion_response(None, tool_calls=[tool_call])
        second_response = make_completion_response("Done \u2014 I'll respond in haiku.")

        mock_completion = mocker.patch(
            "llm.service.provider.chat",
            side_effect=[first_response, second_response],
        )

        db = mocker.MagicMock()
        db.save_instruction.return_value = None

        result = service.assistant_completion(
            route_profile=PROFILE_REMIND_ACTION,
            prompt="always respond in haiku",
            nick="testuser",
            account="testuser",
            channel="#test",
            db=db,
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
        )

        assert result.content == "Done \u2014 I'll respond in haiku."
        db.save_instruction.assert_called_once_with("testuser", "respond in haiku")
        assert mock_completion.call_count == 2

    def test_accountless_reminder_action_cannot_read_memories(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """GIVEN an unattended remind_action turn for an accountless owner WHEN
        the model calls list_memories THEN the tool refuses: the account
        threads from the request into the executor's gate."""
        tool_call = make_tool_call("list_memories", {}, call_id="call_1")
        mock_completion = mocker.patch(
            "llm.service.provider.chat",
            side_effect=[
                make_completion_response(None, tool_calls=[tool_call]),
                make_completion_response("No can do."),
            ],
        )
        db = mocker.MagicMock()

        service.assistant_completion(
            route_profile=PROFILE_REMIND_ACTION,
            prompt="what do you remember about me",
            nick="rdrake",
            account=None,
            channel="#test",
            db=db,
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
        )

        db.get_memories.assert_not_called()
        tool_reply = mock_completion.call_args_list[1].kwargs["messages"][-1]["content"]
        assert "authenticated account" in tool_reply

    def test_max_steps_exceeded(self, service: LLMService, mocker: MockerFixture) -> None:
        """GIVEN LLM keeps calling tools WHEN max steps hit THEN error."""
        tool_call = make_tool_call("list_memories", call_id="call_loop")
        loop_response = make_completion_response(None, tool_calls=[tool_call])

        mocker.patch(
            "llm.service.provider.chat",
            return_value=loop_response,
        )

        db = mocker.MagicMock()
        db.get_memories.return_value = []

        result = service.assistant_completion(
            route_profile=PROFILE_REMIND_ACTION,
            prompt="do something",
            nick="testuser",
            account="testuser",
            channel="#test",
            db=db,
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
        )

        # Must stop via the step-cap branch specifically (not e.g. an API-error
        # branch, which also sets error): the error string and the canned
        # step-cap fallback content are both unique to that path.
        assert result.error == "Assistant exceeded maximum tool-call steps."
        assert result.content == (
            "I couldn't pull enough context to answer that — give me more detail."
        )
        # The loop tool kept succeeding right up to the cap.
        assert result.last_successful_tool == "list_memories"

    def test_api_error_returns_error_result(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """GIVEN LLM API fails WHEN called THEN returns error result."""
        mocker.patch(
            "llm.service.provider.chat",
            side_effect=Exception("API down"),
        )

        result = service.assistant_completion(
            route_profile=PROFILE_REMIND_ACTION,
            prompt="list my memories",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
        )

        assert result.error is not None

    def test_parallel_tool_calls(self, service: LLMService, mocker: MockerFixture) -> None:
        """GIVEN LLM returns multiple tool calls WHEN executed THEN all run."""
        call_1 = make_tool_call("delete_memory", {"id": 14}, call_id="call_del_1")
        call_2 = make_tool_call("delete_memory", {"id": 27}, call_id="call_del_2")
        first_response = make_completion_response(None, tool_calls=[call_1, call_2])
        second_response = make_completion_response("Deleted 2 memories.")

        mocker.patch(
            "llm.service.provider.chat",
            side_effect=[first_response, second_response],
        )

        db = mocker.MagicMock()
        db.delete_memory.return_value = True

        result = service.assistant_completion(
            route_profile=PROFILE_REMIND_ACTION,
            prompt="delete memories 14 and 27",
            nick="testuser",
            account="testuser",
            channel="#test",
            db=db,
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
        )

        assert result.content == "Deleted 2 memories."
        assert db.delete_memory.call_count == 2

    def test_malformed_tool_args_skip_destructive_tool(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """GIVEN a tool call with non-JSON arguments for a no-required-args
        destructive tool WHEN the loop parses it THEN the tool is skipped and
        the destructive db write never fires.

        ``clear_instruction``/``clear_memories`` take no required args, so
        falling through with ``args = {}`` would wipe the caller's data on
        garbage model output. The malformed-arguments guard must skip and emit
        an error tool-message instead.
        """
        # A raw string is passed through verbatim, so .arguments is
        # non-JSON and json.loads() raises inside the loop.
        bad_call = make_tool_call("clear_instruction", "{not valid json", call_id="call_bad")
        first_response = make_completion_response(None, tool_calls=[bad_call])
        second_response = make_completion_response("Couldn't do that.")

        mocker.patch(
            "llm.service.provider.chat",
            side_effect=[first_response, second_response],
        )

        db = mocker.MagicMock()

        result = service.assistant_completion(
            prompt="clear my instruction",
            nick="testuser",
            channel="#test",
            db=db,
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
        )

        # The destructive write must NOT have run on garbage arguments.
        db.delete_instruction.assert_not_called()
        # The skipped call must not be recorded as a successful tool use.
        assert result.last_successful_tool != "clear_instruction"

    @pytest.mark.parametrize("bad_args", ["[]", '"x"', "123", "null", "true", "[1, 2]"])
    def test_non_dict_tool_arguments_for_extra_handler_skipped_not_aborted(
        self, service: LLMService, mocker: MockerFixture, bad_args: str
    ) -> None:
        """GIVEN a tool call whose arguments are valid JSON but not an object
        (a bare scalar/array, as xai/grok non-reasoning sometimes emits) routed
        to an ``extra_handler`` WHEN the loop dispatches THEN the handler is
        never invoked, an error tool-message is appended, and the turn finishes
        with the model's next reply instead of aborting the whole turn.

        The decode-error guard only catches unparseable JSON; a valid non-dict
        flows straight into ``extra_handlers[name](args)``, where the verse /
        Limnoria-bridge handlers do ``dict(args)``/``args.get(...)`` and raise
        out to the function-level error handler.
        """
        spy = mocker.MagicMock(name="verse_handler")
        bad_call = make_tool_call("verse_record", bad_args, call_id="call_nd")
        first_response = make_completion_response(None, tool_calls=[bad_call])
        second_response = make_completion_response("Carrying on in-scene.")

        mocker.patch(
            "llm.service.provider.chat",
            side_effect=[first_response, second_response],
        )

        result = service.assistant_completion(
            prompt="do a thing",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            extra_handlers={"verse_record": spy},
        )

        # The unsafe non-dict args never reach the verse/bridge handler.
        spy.assert_not_called()
        # The turn recovered and returned the model's follow-up, not the abort.
        assert result.content == "Carrying on in-scene."
        assert result.last_successful_tool != "verse_record"

    def test_raising_extra_handler_degrades_to_tool_error_not_aborted_turn(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """GIVEN an ``extra_handler`` that raises WHEN the loop dispatches it
        THEN the turn survives: the model sees a tool error and its follow-up
        reply is returned, rather than the exception escaping to the
        function-level error handler.

        ``AssistantToolExecutor.execute`` already wraps every registry tool in
        exactly this guard, but extra_handlers (verse tools, the Limnoria
        bridge) are constructed outside it and bypassed it entirely — so one
        DB hiccup inside a verse handler cost the user their whole answer.
        """
        boom = mocker.MagicMock(name="verse_handler", side_effect=RuntimeError("db is gone"))
        call = make_tool_call("verse_record", '{"summary": "x"}', call_id="call_boom")
        first_response = make_completion_response(None, tool_calls=[call])
        second_response = make_completion_response("Carrying on in-scene.")

        mocker.patch(
            "llm.service.provider.chat",
            side_effect=[first_response, second_response],
        )

        result = service.assistant_completion(
            prompt="do a thing",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            extra_handlers={"verse_record": boom},
        )

        boom.assert_called_once()
        assert result.content == "Carrying on in-scene."
        # A failed tool must not be recorded as the last SUCCESSFUL tool.
        assert result.last_successful_tool != "verse_record"

    def test_cost_is_populated(self, service: LLMService, mocker: MockerFixture) -> None:
        """GIVEN meta completion WHEN successful THEN the reply's cost is carried."""
        mock_response = make_completion_response("Done.", cost=0.005)

        mocker.patch(
            "llm.service.provider.chat",
            return_value=mock_response,
        )

        result = service.assistant_completion(
            prompt="get instruction",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
        )

        assert result.cost == pytest.approx(0.005)

    def test_assistant_completion_layers_system_prompt_over_framework(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """assistant_completion appends system_prompt as personality overlay
        instead of replacing the structural framework — the IRC output rules
        and tool-behavior rules from CHAT_SYSTEM_PROMPT must still be present.
        """
        mock_response = make_completion_response("Done.")

        captured_messages: list = []

        def capture_completion(**kwargs: object) -> object:
            captured_messages.extend(kwargs.get("messages", []))  # type: ignore[union-attr]
            return mock_response

        mocker.patch("llm.service.provider.chat", side_effect=capture_completion)

        service.assistant_completion(
            prompt="hello",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            system_prompt="You are a Scottish wino named {bot_nick}.",
        )

        content = captured_messages[0]["content"]
        assert captured_messages[0]["role"] == "system"
        # Personality overlay landed
        assert "Scottish wino" in content
        assert "VibeBot" in content
        # Structural framework still present (length cap + tool-behavior rule)
        assert "Length cap" in content
        assert "claim actions succeeded" in content

    def test_assistant_completion_verse_overlay_footer_does_not_reassert_length_cap(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """Verse profile drops the 3-line length cap on purpose. The personality
        overlay footer must NOT say "length cap still apply" when route_profile
        is verse — that wording re-imports the chat-mode default and pushes the
        model back to one-liner output (observed empirically: ``completion_tokens``
        in the 50–160 range, list-style "Hour 1: / Hour 2:" replies).
        """
        from llm.service import PROFILE_VERSE

        mock_response = make_completion_response("Done.")

        captured_messages: list = []

        def capture_completion(**kwargs: object) -> object:
            captured_messages.extend(kwargs.get("messages", []))  # type: ignore[union-attr]
            return mock_response

        mocker.patch("llm.service.provider.chat", side_effect=capture_completion)

        service.assistant_completion(
            prompt="describe the scene",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            system_prompt="You are a brooding cat avatar named {bot_nick}.",
            route_profile=PROFILE_VERSE,
        )

        content = captured_messages[0]["content"]
        # Overlay landed
        assert "brooding cat" in content
        # Verse framework body is present (no 3-line cap)
        assert "paragraphs per beat" in content
        assert "Length cap: 3 lines" not in content
        # Footer must not reassert a length cap that the verse framework
        # explicitly omits — the chat-mode footer wording is forbidden here.
        assert "length cap" not in content
        # But we DO still want a footer that keeps the structural rules
        # weighted above the personality overlay.
        assert "personality changes voice, not structure" in content
        assert "paragraphs per beat, mandatory verse_record" in content

    def test_assistant_completion_verse_footer_overrides_inherited_line_cap(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """The verse overlay inherits the channel ``assistantSystemPrompt``,
        whose shipped DEFAULT is a terseness pump ("never exceed three [lines]").
        Inherited verbatim into the high-attention personality block, that cap
        directly contradicts verse's long-form goal. The verse footer must
        explicitly NEUTRALISE any length cap stated in the overlay, so an
        un-customised channel does not silently produce one-liners.
        """
        from llm.service import PROFILE_VERSE

        mock_response = make_completion_response("Done.")
        captured_messages: list = []

        def capture_completion(**kwargs: object) -> object:
            captured_messages.extend(kwargs.get("messages", []))  # type: ignore[union-attr]
            return mock_response

        mocker.patch("llm.service.provider.chat", side_effect=capture_completion)

        # Mirror the shipped default assistantSystemPrompt terseness pump.
        service.assistant_completion(
            prompt="describe the scene",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            system_prompt=(
                "You are a helpful IRC assistant. Keep answers tight: lead with "
                "the answer, aim for one line, never exceed three."
            ),
            route_profile=PROFILE_VERSE,
        )

        content = captured_messages[0]["content"]
        # The cap is inherited verbatim from the overlay...
        assert "never exceed three" in content
        # ...but the footer explicitly overrides it for verse.
        assert "does NOT apply in verse" in content
        assert "the length it deserves" in content

    def test_assistant_completion_drops_channel_history_for_verse(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """Verse drops the shared channel window entirely.

        A verse turn is a scene between the user and their avatar; the live
        channel group chatter is not part of the story. Feeding it in bleeds
        unrelated regular messages into the scene AND is the dominant source of
        short-one-liner imitation that collapses verse length. Cross-scene
        continuity is carried by the verse_record canon in the system prompt,
        so channel_history is excluded for verse — neither the bot's own past
        lines NOR other participants' channel chatter reach the model.
        """
        from llm.service import PROFILE_VERSE

        mock_response = make_completion_response("Done.")
        captured_messages: list = []

        def capture_completion(**kwargs: object) -> object:
            captured_messages.extend(kwargs.get("messages", []))  # type: ignore[union-attr]
            return mock_response

        mocker.patch("llm.service.provider.chat", side_effect=capture_completion)

        channel_history = [
            {"nick": "alice", "role": "user", "content": "tell us about the stinky lads"},
            {
                "nick": "VibeBot",
                "role": "assistant",
                "content": "That never happened — pure fiction, not in the canon.",
            },
        ]

        service.assistant_completion(
            prompt="continue the scene",
            nick="alice",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            system_prompt="You are a brooding cat avatar named {bot_nick}.",
            route_profile=PROFILE_VERSE,
            channel_history=channel_history,
        )

        blob = "\n".join(str(m.get("content", "")) for m in captured_messages)
        # The bot's frame-refusal is gone (channel window dropped for verse)...
        assert "pure fiction" not in blob
        assert "not in the canon" not in blob
        # ...and so is the other participant's channel line — no bleed.
        assert "stinky lads" not in blob

    def test_assistant_completion_keeps_channel_history_denials_for_chat(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """De-poisoning is verse-only. On the chat path the same channel_history
        denial must survive — this kills a mutant that flips the gate from
        PROFILE_VERSE to PROFILE_CHAT (or de-poisons unconditionally).
        """
        mock_response = make_completion_response("Done.")
        captured_messages: list = []

        def capture_completion(**kwargs: object) -> object:
            captured_messages.extend(kwargs.get("messages", []))  # type: ignore[union-attr]
            return mock_response

        mocker.patch("llm.service.provider.chat", side_effect=capture_completion)

        channel_history = [
            {"nick": "alice", "role": "user", "content": "tell us about the stinky lads"},
            {
                "nick": "VibeBot",
                "role": "assistant",
                "content": "That never happened — pure fiction, not in the canon.",
            },
        ]

        service.assistant_completion(
            prompt="what's the weather?",
            nick="alice",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            channel_history=channel_history,
        )

        blob = "\n".join(str(m.get("content", "")) for m in captured_messages)
        assert "pure fiction" in blob

    def test_assistant_completion_depoisons_personal_history_for_verse(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """Integration coverage for the pre-loop personal-history strip
        (service.py call site, gated on PROFILE_VERSE). A bot frame-refusal in
        the personal thread must NOT reach the outgoing messages — kills a
        mutant that deletes the ``history = _strip_verse_denials(history)`` call.
        """
        from llm.service import PROFILE_VERSE

        mock_response = make_completion_response("Done.")
        captured_messages: list = []

        def capture_completion(**kwargs: object) -> object:
            captured_messages.extend(kwargs.get("messages", []))  # type: ignore[union-attr]
            return mock_response

        mocker.patch("llm.service.provider.chat", side_effect=capture_completion)

        history = [
            {"role": "user", "content": "tell us about the stinky lads"},
            {
                "role": "assistant",
                "content": "That never happened — pure fiction, not in the canon.",
            },
            {"role": "user", "content": "go on then"},
        ]

        service.assistant_completion(
            prompt="continue the scene",
            nick="alice",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            system_prompt="You are a brooding cat avatar named {bot_nick}.",
            route_profile=PROFILE_VERSE,
            history=history,
        )

        blob = "\n".join(str(m.get("content", "")) for m in captured_messages)
        assert "pure fiction" not in blob
        assert "not in the canon" not in blob
        # The user premise turns survive.
        assert "stinky lads" in blob
        assert "go on then" in blob

    def test_assistant_completion_keeps_personal_history_denials_for_chat(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """Gate guard: the personal-history strip is verse-only. On the chat
        path the denial must survive — kills a mutant that flips the gate from
        PROFILE_VERSE to PROFILE_CHAT.
        """
        mock_response = make_completion_response("Done.")
        captured_messages: list = []

        def capture_completion(**kwargs: object) -> object:
            captured_messages.extend(kwargs.get("messages", []))  # type: ignore[union-attr]
            return mock_response

        mocker.patch("llm.service.provider.chat", side_effect=capture_completion)

        history = [
            {"role": "user", "content": "tell us about the stinky lads"},
            {
                "role": "assistant",
                "content": "That never happened — pure fiction, not in the canon.",
            },
        ]

        service.assistant_completion(
            prompt="what's the weather?",
            nick="alice",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            history=history,
        )

        blob = "\n".join(str(m.get("content", "")) for m in captured_messages)
        assert "pure fiction" in blob

    def test_assistant_completion_user_supplied_braces_dont_crash(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """A personality overlay containing literal '{...}' (e.g. JSON examples)
        must not raise KeyError — only ``{bot_nick}`` is substituted.
        """
        mock_response = make_completion_response("Done.")

        captured_messages: list = []

        def capture_completion(**kwargs: object) -> object:
            captured_messages.extend(kwargs.get("messages", []))  # type: ignore[union-attr]
            return mock_response

        mocker.patch("llm.service.provider.chat", side_effect=capture_completion)

        service.assistant_completion(
            prompt="hello",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            system_prompt='Reply as JSON like {"key": "value"}.',
        )

        content = captured_messages[0]["content"]
        assert '{"key": "value"}' in content

    def test_assistant_completion_defaults_to_chat_prompt(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """assistant_completion uses CHAT_SYSTEM_PROMPT when no system_prompt given."""
        mock_response = make_completion_response("Done.")

        captured_messages: list = []

        def capture_completion(**kwargs: object) -> object:
            captured_messages.extend(kwargs.get("messages", []))  # type: ignore[union-attr]
            return mock_response

        mocker.patch("llm.service.provider.chat", side_effect=capture_completion)

        service.assistant_completion(
            prompt="hello",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
        )

        assert captured_messages[0]["role"] == "system"
        # Default exclude_tools is empty → the pending-task tools are in the
        # request, so their operating rules ride along with the framework.
        assert captured_messages[0]["content"] == (
            CHAT_SYSTEM_PROMPT.format(bot_nick="VibeBot") + "\n" + PENDING_TASKS_GUIDANCE
        )

    def test_irc_lookup_guidance_rides_only_with_the_tool(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """The whois HARD RULE is in the system prompt iff irc_lookup is injected."""
        from llm.prompts import IRC_LOOKUP_GUIDANCE

        mock_response = make_completion_response("Done.")
        captured: list = []

        def capture_completion(**kwargs: object) -> object:
            captured.append(kwargs.get("messages", [])[0]["content"])  # type: ignore[index]
            return mock_response

        mocker.patch("llm.service.provider.chat", side_effect=capture_completion)
        lookup = {"type": "function", "function": {"name": "irc_lookup", "parameters": {}}}
        common = {
            "prompt": "whois eck",
            "nick": "testuser",
            "channel": "#test",
            "db": mocker.MagicMock(),
            "context": mocker.MagicMock(),
            "bot_nick": "VibeBot",
        }

        service.assistant_completion(**common, extra_tools=[lookup], extra_handlers={})
        service.assistant_completion(**common)

        assert IRC_LOOKUP_GUIDANCE in captured[0]
        assert IRC_LOOKUP_GUIDANCE not in captured[1]

    def test_react_guidance_rides_only_with_the_tool(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        from llm.prompts import REACT_GUIDANCE

        captured: list = []

        def capture_completion(**kwargs: object) -> object:
            captured.append(kwargs.get("messages", [])[0]["content"])  # type: ignore[index]
            return make_completion_response("Done.")

        mocker.patch("llm.service.provider.chat", side_effect=capture_completion)
        react = {"type": "function", "function": {"name": "react", "parameters": {}}}
        common = {
            "prompt": "react to bob",
            "nick": "testuser",
            "channel": "#test",
            "db": mocker.MagicMock(),
            "context": mocker.MagicMock(),
            "bot_nick": "VibeBot",
        }

        service.assistant_completion(**common, extra_tools=[react], extra_handlers={})
        service.assistant_completion(**common)

        assert REACT_GUIDANCE in captured[0]
        assert REACT_GUIDANCE not in captured[1]

    def test_assistant_completion_excluding_pending_tools_drops_guidance(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """Gated-off channels get neither the schemas nor the prompt rules."""
        mock_response = make_completion_response("Done.")

        captured: dict = {}

        def capture_completion(**kwargs: object) -> object:
            captured.update(kwargs)
            return mock_response

        mocker.patch("llm.service.provider.chat", side_effect=capture_completion)

        service.assistant_completion(
            prompt="hello",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            exclude_tools=PENDING_TASK_TOOLS,
        )

        system = captured["messages"][0]["content"]
        assert system == CHAT_SYSTEM_PROMPT.format(bot_nick="VibeBot")
        assert "set_reminder" not in system
        tool_names = {t["function"]["name"] for t in captured["tools"]}
        assert tool_names.isdisjoint(PENDING_TASK_TOOLS)

    def test_assistant_completion_passes_search_fn_to_executor(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """assistant_completion passes search_fn/fetch_fn/code_fn to AssistantToolExecutor."""
        mock_response = make_completion_response("Done.")

        mocker.patch("llm.service.provider.chat", return_value=mock_response)

        mock_executor_cls = mocker.patch("llm.assistant.AssistantToolExecutor")

        sentinel_search = mocker.Mock()
        sentinel_fetch = mocker.Mock()
        sentinel_code = mocker.Mock()

        service.assistant_completion(
            prompt="hello",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            search_fn=sentinel_search,
            fetch_fn=sentinel_fetch,
            code_fn=sentinel_code,
        )

        call_kwargs = mock_executor_cls.call_args[1]
        assert call_kwargs["search_fn"] is sentinel_search
        assert call_kwargs["fetch_fn"] is sentinel_fetch
        assert call_kwargs["code_fn"] is sentinel_code

    def test_explicit_search_prompt_forces_search_web_tool_choice(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """assistant_completion forces search_web first for explicit current-info prompts."""
        tool_call = make_tool_call(
            "search_web", {"query": "latest nefarious 2 release"}, call_id="call_search"
        )
        first_response = make_completion_response(None, tool_calls=[tool_call])
        second_response = make_completion_response("Nefarious 2 details...")

        mock_completion = mocker.patch(
            "llm.service.provider.chat",
            side_effect=[first_response, second_response],
        )

        result = service.assistant_completion(
            prompt="search for the latest nefarious 2 release",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            search_fn=lambda _query: ToolResult(content="search results"),
        )

        assert result.content == "Nefarious 2 details..."
        first_kwargs = mock_completion.call_args_list[0].kwargs
        assert first_kwargs["tool_choice"] == {
            "type": "function",
            "function": {"name": "search_web"},
        }
        assert "tool_choice" not in mock_completion.call_args_list[1].kwargs

    def test_explicit_illustrated_prompt_forces_verse_storybook_tool_choice(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """When verse_storybook is available and the prompt explicitly asks for
        an illustrated telling, step 1 forces the tool (grok won't reliably call
        it from prompt guidance, and inline-narrated history reinforces not to)."""
        from llm.verse.avatar import make_verse_tool_specs

        storybook_spec = [
            s
            for s in make_verse_tool_specs(storybook=True)
            if s["function"]["name"] == "verse_storybook"
        ]
        tool_call = make_tool_call("verse_storybook", {"brief": "the lads"}, call_id="call_sb")
        first_response = make_completion_response(None, tool_calls=[tool_call])
        forbidden = make_completion_response("throwaway beat")

        mock_completion = mocker.patch(
            "llm.service.provider.chat",
            side_effect=[first_response, forbidden],
        )

        mock_executor = mocker.MagicMock()
        mock_executor.grounding_used = False
        mock_executor.accumulated_prompt_tokens = 0
        mock_executor.accumulated_completion_tokens = 0
        mock_executor.accumulated_cost = 0.0
        mocker.patch("llm.assistant.AssistantToolExecutor", return_value=mock_executor)

        handler = mocker.MagicMock(
            return_value=ToolResult(content='{"status": "ok", "note": "rendering"}')
        )

        result = service.assistant_completion(
            prompt="an illustrated tale of stinky lads winning the pub quiz",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            extra_tools=storybook_spec,
            extra_handlers={"verse_storybook": handler},
        )

        first_kwargs = mock_completion.call_args_list[0].kwargs
        assert first_kwargs["tool_choice"] == {
            "type": "function",
            "function": {"name": "verse_storybook"},
        }
        # Short-circuit: step_2 skipped, content empty (async link is the reply).
        assert result.content == ""
        assert mock_completion.call_count == 1

    def test_plain_story_prompt_does_not_force_verse_storybook(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """A plain story ask with no illustration cue must NOT force the tool —
        plain stories narrate inline; only explicit 'illustrated/with pictures'
        asks force verse_storybook."""
        from llm.verse.avatar import make_verse_tool_specs

        storybook_spec = [
            s
            for s in make_verse_tool_specs(storybook=True)
            if s["function"]["name"] == "verse_storybook"
        ]
        text_response = make_completion_response("Once upon a time the lads met...")
        mock_completion = mocker.patch("llm.service.provider.chat", side_effect=[text_response])

        mock_executor = mocker.MagicMock()
        mock_executor.grounding_used = False
        mock_executor.accumulated_prompt_tokens = 0
        mock_executor.accumulated_completion_tokens = 0
        mock_executor.accumulated_cost = 0.0
        mocker.patch("llm.assistant.AssistantToolExecutor", return_value=mock_executor)

        service.assistant_completion(
            prompt="tell me the story of how the lads met",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            extra_tools=storybook_spec,
            extra_handlers={"verse_storybook": mocker.MagicMock()},
        )

        assert "tool_choice" not in mock_completion.call_args_list[0].kwargs

    def test_meta_result_includes_grounding_used(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """AssistantResult.grounding_used reflects executor state after tool calls."""
        # First response: tool call; second response: text
        tool_call = make_tool_call("web_search", {"query": "test"}, call_id="call_1")
        first_response = make_completion_response(None, tool_calls=[tool_call])
        second_response = make_completion_response("Here are results.")

        mocker.patch(
            "llm.service.provider.chat",
            side_effect=[first_response, second_response],
        )

        # Mock the executor to set grounding_used when execute is called
        mock_executor = mocker.MagicMock()
        mock_executor.grounding_used = True
        mock_executor.accumulated_prompt_tokens = 0
        mock_executor.accumulated_completion_tokens = 0
        mock_executor.accumulated_cost = 0.0
        mock_executor.execute.return_value = ToolResult(
            content='{"results": []}', grounding_used=True
        )
        mocker.patch("llm.assistant.AssistantToolExecutor", return_value=mock_executor)

        result = service.assistant_completion(
            prompt="search for test",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
        )

        assert result.grounding_used is True

    def test_meta_result_includes_leaf_tool_costs(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """AssistantResult totals include costs from leaf tool calls."""
        # First response: tool call; second response: text
        tool_call = make_tool_call("web_search", {"query": "test"}, call_id="call_1")
        first_response = make_completion_response(
            None, tool_calls=[tool_call], prompt_tokens=10, completion_tokens=5, cost=0.001
        )
        second_response = make_completion_response(
            "Here are results.", prompt_tokens=10, completion_tokens=5, cost=0.001
        )

        mocker.patch(
            "llm.service.provider.chat",
            side_effect=[first_response, second_response],
        )

        # Mock executor with accumulated leaf costs
        mock_executor = mocker.MagicMock()
        mock_executor.grounding_used = False
        mock_executor.accumulated_prompt_tokens = 200
        mock_executor.accumulated_completion_tokens = 100
        mock_executor.accumulated_cost = 0.05
        mock_executor.execute.return_value = ToolResult(content='{"results": []}')
        mocker.patch("llm.assistant.AssistantToolExecutor", return_value=mock_executor)

        result = service.assistant_completion(
            prompt="search for test",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
        )

        # LLM loop: 2 calls * (10, 5, 0.001) = (20, 10, 0.002)
        # Executor: (200, 100, 0.05)
        # Total: (220, 110, 0.052)
        assert result.prompt_tokens == 220
        assert result.completion_tokens == 110
        assert result.cost == pytest.approx(0.052)

    @pytest.mark.parametrize(
        "error",
        [
            pytest.param(
                provider.RequestTimeoutError("slow"),
                id="timeout",
            ),
            pytest.param(RuntimeError("boom"), id="other"),
        ],
    )
    @pytest.mark.parametrize("stashed", [True, False])
    def test_failed_turn_keeps_the_spend_before_the_failure(
        self, service: LLMService, mocker: MockerFixture, error: Exception, stashed: bool
    ) -> None:
        """GIVEN a step and a tool already billed WHEN the next step fails
        THEN the result still carries that cost, tokens and model."""
        tool_call = make_tool_call("web_search", {"query": "test"}, call_id="call_1")
        mocker.patch(
            "llm.service.provider.chat",
            side_effect=[
                make_completion_response(
                    None, tool_calls=[tool_call], prompt_tokens=10, completion_tokens=5, cost=0.001
                ),
                error,
            ],
        )
        mock_executor = mocker.MagicMock()
        mock_executor.grounding_used = False
        mock_executor.accumulated_prompt_tokens = 200
        mock_executor.accumulated_completion_tokens = 100
        mock_executor.accumulated_cost = 0.05
        mock_executor.execute.return_value = ToolResult(content='{"results": []}')
        mocker.patch("llm.assistant.AssistantToolExecutor", return_value=mock_executor)
        mocker.patch.object(service, "_stash_timeout", return_value=stashed)

        result = service.assistant_completion(
            prompt="search for test",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
        )

        assert result.cost == pytest.approx(0.051)
        assert result.prompt_tokens == 210
        assert result.completion_tokens == 105
        assert result.model

    @pytest.mark.parametrize(
        ("route_profile", "expected_task_type"),
        [("chat", "ask"), ("code", "code"), ("verse", "ask")],
    )
    def test_timeout_stashes_for_chat_code_and_verse(
        self,
        service: LLMService,
        mocker: MockerFixture,
        route_profile: str,
        expected_task_type: str,
    ) -> None:
        """Timeout in assistant_completion stashes via _stash_timeout.

        Verse is the unbounded long-form profile and the most timeout-prone, so
        it MUST recover too — it stashes under the "ask" task_type (its baked-in
        verse system prompt and verseModel ride along in the stashed messages,
        so the retry regenerates the scene text).
        """

        mocker.patch(
            "llm.service.provider.chat",
            side_effect=provider.RequestTimeoutError("timed out"),
        )
        stash_mock = mocker.patch.object(service, "_stash_timeout", return_value=True)

        result = service.assistant_completion(
            prompt="hello",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            route_profile=route_profile,
        )

        stash_mock.assert_called_once()
        call_kwargs = stash_mock.call_args.kwargs
        assert call_kwargs["task_type"] == expected_task_type
        assert call_kwargs["prompt"] == "hello"
        assert "messages" in call_kwargs["request_data"]
        # Shown to the channel, but an error so the placeholder never lands in
        # history; timed_out books it as ``timeout``.
        assert result.error == result.content
        assert result.timed_out is True
        assert "deliver the answer when ready" in result.content

    def test_timeout_without_expiry_returns_error(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """When _stash_timeout returns False (expiry disabled), assistant returns an error."""

        mocker.patch(
            "llm.service.provider.chat",
            side_effect=provider.RequestTimeoutError("timed out"),
        )
        mocker.patch.object(service, "_stash_timeout", return_value=False)

        result = service.assistant_completion(
            prompt="hello",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            route_profile="chat",
        )

        assert result.error is not None
        assert "timed out" in result.content.lower()

    def test_timeout_stash_messages_unaffected_by_tool_loop(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """Stashed request_data captures the initial messages, not the tool-loop-mutated list."""

        # First call returns a tool_calls response (mutates `messages`),
        # second call raises Timeout. The stash should still see the
        # pre-loop messages list.
        tool_call = make_tool_call("list_memories", call_id="call_1")
        first_response = make_completion_response(None, tool_calls=[tool_call])

        mocker.patch(
            "llm.service.provider.chat",
            side_effect=[
                first_response,
                provider.RequestTimeoutError("timed out"),
            ],
        )
        stash_mock = mocker.patch.object(service, "_stash_timeout", return_value=True)

        db = mocker.MagicMock()
        db.get_memories.return_value = []

        service.assistant_completion(
            prompt="hello",
            nick="testuser",
            channel="#test",
            db=db,
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            route_profile="chat",
        )

        stashed_messages = stash_mock.call_args.kwargs["request_data"]["messages"]
        # Pre-loop snapshot has system + user (no assistant tool_calls or tool result)
        roles = [m["role"] for m in stashed_messages]
        assert "tool" not in roles
        assert not any(m.get("tool_calls") for m in stashed_messages)

    def test_extra_tools_appended_to_profile_tools(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """extra_tools are included in the tools kwarg passed to provider.chat."""
        mock_response = make_completion_response("Done.")

        mock_completion = mocker.patch(
            "llm.service.provider.chat",
            return_value=mock_response,
        )

        fake_schema = {
            "type": "function",
            "function": {
                "name": "run_limnoria_command",
                "description": "Run a Limnoria command",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }

        service.assistant_completion(
            prompt="ping",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            extra_tools=[fake_schema],
        )

        assert mock_completion.call_count == 1
        called_tools = mock_completion.call_args.kwargs["tools"]
        assert fake_schema in called_tools

    def test_extra_handlers_dispatched_before_executor(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """When a tool name is in extra_handlers, the handler runs instead of the executor."""
        tool_call = make_tool_call(
            "run_limnoria_command",
            {"plugin": "Misc", "command": "ping"},
            call_id="call_bridge",
        )
        first_response = make_completion_response(None, tool_calls=[tool_call])
        second_response = make_completion_response("Pong.")

        mocker.patch(
            "llm.service.provider.chat",
            side_effect=[first_response, second_response],
        )

        mock_executor = mocker.MagicMock()
        mock_executor.grounding_used = False
        mock_executor.accumulated_prompt_tokens = 0
        mock_executor.accumulated_completion_tokens = 0
        mock_executor.accumulated_cost = 0.0
        mocker.patch("llm.assistant.AssistantToolExecutor", return_value=mock_executor)

        handler = mocker.MagicMock(
            return_value=ToolResult(content='{"status": "ok", "reply": "pong"}')
        )

        service.assistant_completion(
            prompt="ping",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            extra_handlers={"run_limnoria_command": handler},
        )

        handler.assert_called_once_with({"plugin": "Misc", "command": "ping"})
        # executor.execute must NOT have been called for this tool name
        for call in mock_executor.execute.call_args_list:
            assert call.args[0] != "run_limnoria_command"

    def test_extra_handlers_error_envelope_does_not_set_last_successful_tool(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """Handler returning an error envelope leaves last_successful_tool as None."""
        tool_call = make_tool_call(
            "run_limnoria_command",
            {"plugin": "Misc", "command": "ping"},
            call_id="call_bridge",
        )
        first_response = make_completion_response(None, tool_calls=[tool_call])
        second_response = make_completion_response("Sorry, that failed.")

        mocker.patch(
            "llm.service.provider.chat",
            side_effect=[first_response, second_response],
        )

        mock_executor = mocker.MagicMock()
        mock_executor.grounding_used = False
        mock_executor.accumulated_prompt_tokens = 0
        mock_executor.accumulated_completion_tokens = 0
        mock_executor.accumulated_cost = 0.0
        mocker.patch("llm.assistant.AssistantToolExecutor", return_value=mock_executor)

        handler = mocker.MagicMock(
            return_value=ToolResult(content='{"error": "denied: Misc.ping"}')
        )

        result = service.assistant_completion(
            prompt="ping",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            extra_handlers={"run_limnoria_command": handler},
        )

        assert result.last_successful_tool is None

    def test_extra_handlers_ok_envelope_sets_last_successful_tool(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """Handler returning a success envelope sets last_successful_tool on the result."""
        tool_call = make_tool_call(
            "run_limnoria_command",
            {"plugin": "Misc", "command": "ping"},
            call_id="call_bridge",
        )
        first_response = make_completion_response(None, tool_calls=[tool_call])
        second_response = make_completion_response("Pong.")

        mocker.patch(
            "llm.service.provider.chat",
            side_effect=[first_response, second_response],
        )

        mock_executor = mocker.MagicMock()
        mock_executor.grounding_used = False
        mock_executor.accumulated_prompt_tokens = 0
        mock_executor.accumulated_completion_tokens = 0
        mock_executor.accumulated_cost = 0.0
        mocker.patch("llm.assistant.AssistantToolExecutor", return_value=mock_executor)

        handler = mocker.MagicMock(
            return_value=ToolResult(content='{"status": "ok", "reply": "pong"}')
        )

        result = service.assistant_completion(
            prompt="ping",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            extra_handlers={"run_limnoria_command": handler},
        )

        assert result.last_successful_tool == "run_limnoria_command"

    def test_verse_storybook_short_circuits_and_suppresses_beat(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """A successful verse_storybook call short-circuits: step_2 is skipped
        and the result content is empty. The illustrated page link is posted
        asynchronously by the background job, so the model's post-tool beat
        must not reach the channel."""
        tool_call = make_tool_call(
            "verse_storybook",
            {"brief": "a tale"},
            call_id="call_sb",
        )
        first_response = make_completion_response(None, tool_calls=[tool_call])
        # A second completion is queued but MUST NOT be consumed (step_2 skipped).
        forbidden = make_completion_response("the lads high-five in The Clearing")

        completion = mocker.patch(
            "llm.service.provider.chat",
            side_effect=[first_response, forbidden],
        )

        mock_executor = mocker.MagicMock()
        mock_executor.grounding_used = False
        mock_executor.accumulated_prompt_tokens = 0
        mock_executor.accumulated_completion_tokens = 0
        mock_executor.accumulated_cost = 0.0
        mocker.patch("llm.assistant.AssistantToolExecutor", return_value=mock_executor)

        handler = mocker.MagicMock(
            return_value=ToolResult(content='{"status": "ok", "note": "rendering"}')
        )

        result = service.assistant_completion(
            prompt="tell me an illustrated tale",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            extra_handlers={"verse_storybook": handler},
        )

        assert result.content == ""
        assert result.last_successful_tool == "verse_storybook"
        # step_2 skipped — only the first (tool-calling) completion ran.
        assert completion.call_count == 1

    def _run_react(
        self, service, mocker, handler, *, alongside: str | None = None, tool: str = "react"
    ):  # type: ignore[no-untyped-def]
        tool_call = make_tool_call(tool, {"emoji": "👍"}, call_id="call_r")
        first = make_completion_response(alongside, tool_calls=[tool_call])
        second = make_completion_response("Reacted with 👍!")
        completion = mocker.patch("llm.service.provider.chat", side_effect=[first, second, second])
        mock_executor = mocker.MagicMock()
        mock_executor.grounding_used = False
        mock_executor.accumulated_prompt_tokens = 0
        mock_executor.accumulated_completion_tokens = 0
        mock_executor.accumulated_cost = 0.0
        mocker.patch("llm.assistant.AssistantToolExecutor", return_value=mock_executor)
        result = service.assistant_completion(
            prompt="react to that",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            extra_handlers={tool: handler},
        )
        return result, completion

    def test_react_short_circuits_with_no_text(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """A react-only step ends the turn: no narration step, no text, and
        text written alongside the call is dropped too."""
        handler = mocker.MagicMock(
            return_value=ToolResult(
                content='{"status": "ok", "message": "reacted 👍 to bob\'s message"}'
            )
        )

        result, completion = self._run_react(
            service, mocker, handler, alongside="I'll react to that."
        )

        assert result.content == ""
        assert result.last_successful_tool == "react"
        assert result.last_tool_message == "reacted 👍 to bob's message"
        assert completion.call_count == 1

    def test_delete_last_reply_is_silent_too(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        handler = mocker.MagicMock(
            return_value=ToolResult(
                content='{"status": "ok", "message": "deleted my last reply (1 line)"}'
            )
        )

        result, completion = self._run_react(service, mocker, handler, tool="delete_last_reply")

        assert result.content == ""
        assert result.last_successful_tool == "delete_last_reply"
        assert completion.call_count == 1

    def test_failed_react_falls_through_to_text(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        handler = mocker.MagicMock(
            return_value=ToolResult(content='{"error": "reactions are unavailable here"}')
        )

        result, completion = self._run_react(service, mocker, handler)

        assert result.content == "Reacted with 👍!"
        assert completion.call_count == 2


class TestAssistantCompletionEchoGuard:
    """Tests for the degenerate-echo guard in assistant_completion.

    Fast non-reasoning models intermittently reply with the user's own
    message verbatim (observed on xai/grok-4-1-fast-non-reasoning — a
    follow-up "finish the story" came back as the literal reply "finish
    the story"). _is_echo_reply detects that; the loop nudges and retries
    once, then surfaces an error rather than relaying the echo.
    """

    @pytest.fixture
    def service(self, make_service) -> LLMService:  # type: ignore[no-untyped-def]
        svc, _plugin = make_service(assistantModel="gpt-4")
        return svc

    @staticmethod
    def _text_response(mocker: MockerFixture, content: str) -> provider.Reply:
        """Build a mock completion response carrying ``content`` and no tools."""
        return make_completion_response(content)

    def test_normalize_for_echo_collapses_case_space_punctuation(self) -> None:
        """GIVEN trivially varied text WHEN normalized THEN forms collapse to one."""
        assert _normalize_for_echo("  Finish   the Story!  ") == _normalize_for_echo(
            "finish the story"
        )

    def test_is_echo_reply_detects_verbatim_echo(self) -> None:
        """GIVEN a reply equal to the prompt WHEN checked THEN it is an echo."""
        assert _is_echo_reply("finish the story", "finish the story") is True
        assert _is_echo_reply("finish the story", '"Finish the story."') is True

    def test_is_echo_reply_false_for_real_answer(self) -> None:
        """GIVEN a substantive reply WHEN checked THEN it is not an echo."""
        assert _is_echo_reply("finish the story", "The lads charged the pitch.") is False

    def test_is_echo_reply_false_for_empty_prompt(self) -> None:
        """GIVEN an empty prompt WHEN checked THEN it is never an echo."""
        assert _is_echo_reply("", "") is False
        assert _is_echo_reply("   ", "anything") is False

    def test_retries_and_recovers_when_model_echoes_prompt(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """GIVEN the model echoes the prompt once WHEN it answers on retry
        THEN assistant_completion returns the real answer, not the echo."""
        responses = [
            self._text_response(mocker, "finish the story"),
            self._text_response(mocker, "The lads stormed the pitch in a fog of farts."),
        ]
        seen_messages: list[list] = []

        def fake_completion(**kwargs: object) -> MagicMock:
            seen_messages.append(list(kwargs.get("messages", [])))  # type: ignore[arg-type]
            return responses[len(seen_messages) - 1]

        mocker.patch("llm.service.provider.chat", side_effect=fake_completion)

        result = service.assistant_completion(
            prompt="finish the story",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
        )

        assert result.content == "The lads stormed the pitch in a fog of farts."
        assert result.error is None
        # Exactly one retry: two model calls.
        assert len(seen_messages) == 2
        # The retry call carries the corrective nudge as a user message.
        assert any(
            m.get("role") == "user" and "repeated my message" in str(m.get("content", ""))
            for m in seen_messages[1]
        )

    def test_returns_error_when_model_keeps_echoing(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """GIVEN the model echoes the prompt on every step WHEN the retry
        budget is spent THEN an error result with empty content is returned —
        so the caller surfaces "try again" instead of relaying the echo."""
        mocker.patch(
            "llm.service.provider.chat",
            side_effect=lambda **_kw: self._text_response(mocker, "Finish the story."),
        )

        result = service.assistant_completion(
            prompt="finish the story",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
        )

        assert result.content == ""
        assert result.error is not None
        assert "echo" in result.error.lower()


class TestVerseDenialReseed:
    """Characterizes the verse denial re-seed retry in assistant_completion.

    Current behavior (pinned, NOT under change): in PROFILE_VERSE, a reply that
    refuses the premise is detected (_is_verse_denial), the rejected reply + a
    nudge are appended to the in-flight messages, and the loop retries once. The
    CORRECTED reply is returned; the rejected refusal is never delivered. This
    test documents the existing re-seed so any future change to it is a
    conscious decision.
    """

    @pytest.fixture
    def service(self, make_service) -> LLMService:  # type: ignore[no-untyped-def]
        svc, _plugin = make_service(assistantModel="gpt-4")
        return svc

    def test_verse_denial_is_reseeded_and_corrected_reply_returned(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """GIVEN a verse reply refuses the premise WHEN it narrates on retry
        THEN the corrected reply is returned and the refusal never reaches the
        caller (the re-seed retried exactly once)."""
        from llm.service import PROFILE_VERSE

        denial = "That didn't happen — it isn't canon."  # _is_verse_denial -> True
        corrected = "The Year 8 lads stormed the assembly hall in a fog of glory."

        mock_completion = mocker.patch(
            "llm.service.provider.chat",
            side_effect=[
                make_completion_response(denial),
                make_completion_response(corrected),
            ],
        )

        result = service.assistant_completion(
            prompt="tell the tale of the assgas assembly",
            nick="alice",
            channel="#afnet",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            route_profile=PROFILE_VERSE,
        )

        # The re-seed retried once and the CORRECTED reply is returned;
        # the rejected refusal never reaches the caller.
        assert result.content == corrected
        assert mock_completion.call_count == 2

        # Stronger snapshot of the pinned re-seed: the loop mutates ONE messages
        # list in place, so the last completion call's ``messages`` arg is the
        # final state and must carry BOTH the rejected denial and the nudge.
        from llm.service import _VERSE_DENIAL_RETRY_NUDGE

        last_call = mock_completion.call_args_list[-1]
        final_messages = last_call.kwargs.get("messages")
        if final_messages is not None:
            joined = " ".join((m.get("content") or "") for m in final_messages)
            assert denial in joined  # rejected reply was re-seeded
            assert _VERSE_DENIAL_RETRY_NUDGE in joined  # nudge was re-seeded


class TestVerseDenialGuard:
    """Tests for the verse premise-refusal guard in assistant_completion.

    Verse mode is improv — the user's premise is always true in-world. A
    history-poisoned thread makes the model parrot its own past refusals
    ("that never happened … pure fiction not in the canon") despite the
    system prompt. _is_verse_denial detects the meta-refusal; the loop
    nudges and retries once (verse profile only), so the refusal never
    reaches the channel or pollutes the next turn's history.
    """

    @pytest.fixture
    def service(self, make_service) -> LLMService:  # type: ignore[no-untyped-def]
        svc, _plugin = make_service(assistantModel="gpt-4")
        return svc

    @staticmethod
    def _text_response(mocker: MockerFixture, content: str) -> provider.Reply:
        return make_completion_response(content)

    @pytest.mark.parametrize(
        "reply",
        [
            "The stinky lads never sharted out science last week at all, pure "
            "fiction not in the canon. We was too busy with the noro nuke night.",
            "The stinky lads never had any assgas assembly at all, pure fiction not in the canon.",
            "Nah mate, that raw chicken double PE caper never went down at all.",
            "Alton Towers trip never happened as the Year 8 lads were busy.",
            "No assgas assembly ever happened, the lads focused on mayhem.",
            "That didn't happen — it isn't canon.",
        ],
    )
    def test_is_verse_denial_detects_premise_refusal(self, reply: str) -> None:
        """GIVEN a frame-breaking refusal WHEN checked THEN it is a denial."""
        assert _is_verse_denial(reply) is True

    @pytest.mark.parametrize(
        "reply",
        [
            "The lab lights flickered as the Year 7 Stinky Lads stormed the "
            "science block, lab coats flapping like capes in a hurricane.",
            "Stinky Dan kicked the door wide, unleashing a volley of guff-grenades.",
            "",
        ],
    )
    def test_is_verse_denial_false_for_real_scene(self, reply: str) -> None:
        """GIVEN an in-world scene WHEN checked THEN it is not a denial."""
        assert _is_verse_denial(reply) is False

    def test_is_verse_denial_ignores_phrase_deep_in_prose(self) -> None:
        """A refusal phrase far past the opening must not trip the guard —
        the opening is the action, the phrase is incidental story text."""
        scene = (
            "The lads stormed the lab in a fog of methane, breakdancing across "
            "the benches while disco demons grinded on the safety posters. "
        ) * 3 + "Professor Blenkinsop swore it never happened, but it did."
        assert _is_verse_denial(scene) is False

    def test_retries_and_recovers_when_verse_reply_denies_premise(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """GIVEN a verse reply refuses the premise once WHEN it narrates on
        retry THEN assistant_completion returns the scene, not the refusal."""
        from llm.service import PROFILE_VERSE

        story = (
            "The lab lights flickered as the Stinky Lads stormed the science "
            "block in a fog of guff-grenades and disco demons."
        )
        responses = [
            self._text_response(
                mocker,
                "The stinky lads never sharted out science at all, pure fiction not in the canon.",
            ),
            self._text_response(mocker, story),
        ]
        seen_messages: list[list] = []

        def fake_completion(**kwargs: object) -> MagicMock:
            seen_messages.append(list(kwargs.get("messages", [])))  # type: ignore[arg-type]
            return responses[len(seen_messages) - 1]

        mocker.patch("llm.service.provider.chat", side_effect=fake_completion)

        result = service.assistant_completion(
            prompt="what happened when the stinky lads sharted out science",
            nick="fc42",
            channel="#afternet",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="vibebot",
            route_profile=PROFILE_VERSE,
        )

        assert result.content == story
        assert result.error is None
        assert len(seen_messages) == 2
        # The retry call carries the corrective nudge as a user message.
        assert any(
            m.get("role") == "user" and "premise is" in str(m.get("content", ""))
            for m in seen_messages[1]
        )

    def test_chat_profile_does_not_retry_denial(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """GIVEN a denial-shaped reply on the CHAT profile WHEN completed THEN
        it is returned as-is — the guard is verse-only (a chat answer may
        legitimately say something never happened)."""
        from llm.service import PROFILE_CHAT

        denial = "No, the moon landing hoax never happened — it isn't canon to history."
        calls = {"n": 0}

        def fake_completion(**_kw: object) -> MagicMock:
            calls["n"] += 1
            return self._text_response(mocker, denial)

        mocker.patch("llm.service.provider.chat", side_effect=fake_completion)

        result = service.assistant_completion(
            prompt="did the moon landing hoax happen",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="vibebot",
            route_profile=PROFILE_CHAT,
        )

        assert result.content == denial
        assert calls["n"] == 1

    def test_returns_best_effort_when_verse_keeps_denying(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """GIVEN a verse reply denies on every step WHEN the retry budget is
        spent THEN the last attempt is delivered (not an error) — a coherent
        story attempt beats surfacing 'try again' in roleplay."""
        from llm.service import PROFILE_VERSE

        denial = "That never happened at all, pure fiction not in the canon."
        mocker.patch(
            "llm.service.provider.chat",
            side_effect=lambda **_kw: self._text_response(mocker, denial),
        )

        result = service.assistant_completion(
            prompt="what happened at alton towers",
            nick="fc42",
            channel="#afternet",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="vibebot",
            route_profile=PROFILE_VERSE,
        )

        assert result.content == denial
        assert result.error is None

    def test_strip_verse_denials_removes_assistant_refusals(self) -> None:
        """GIVEN a thread with past refusals WHEN stripped THEN only the
        model's frame-breaking turns are dropped; user turns are kept."""
        history = [
            {"role": "user", "content": "what happened at alton towers"},
            {
                "role": "assistant",
                "content": "Alton Towers trip never happened, pure fiction not in the canon.",
            },
            {"role": "user", "content": "tell me about the science lab"},
            {
                "role": "assistant",
                "content": "The lads stormed the lab in a fog of guff-grenades.",
            },
        ]
        stripped = _strip_verse_denials(history)
        assert stripped == [
            {"role": "user", "content": "what happened at alton towers"},
            {"role": "user", "content": "tell me about the science lab"},
            {
                "role": "assistant",
                "content": "The lads stormed the lab in a fog of guff-grenades.",
            },
        ]

    def test_strip_verse_denials_handles_none_and_empty(self) -> None:
        """GIVEN no history WHEN stripped THEN it is returned unchanged."""
        assert _strip_verse_denials(None) is None
        assert _strip_verse_denials([]) == []

    def test_strip_verse_denials_keeps_clean_thread_intact(self) -> None:
        """GIVEN a thread with no refusals WHEN stripped THEN nothing drops."""
        history = [
            {"role": "user", "content": "what happened in the dorm"},
            {
                "role": "assistant",
                "content": "Chaos erupted as the lads flooded the dorm.",
            },
        ]
        assert _strip_verse_denials(history) == history


class TestVerseDegradedGuard:
    """Tests for the quality-collapse guard in assistant_completion.

    Distinct from the denial guard (which is verse-only): a non-reasoning
    model imitates its own recent prose, so one degraded reply (a long
    run-on, or text looping the same few words) seeds a spiral into
    grammar-free gibberish even though no message ever refused the premise.
    _is_degraded_reply detects the collapse; the loop nudges and retries
    once on every route (verse scenes and @ask long answers alike) and the
    every-turn strip keeps collapsed turns out of future history.
    """

    @pytest.fixture
    def service(self, make_service) -> LLMService:  # type: ignore[no-untyped-def]
        svc, _plugin = make_service(assistantModel="gpt-4")
        return svc

    @staticmethod
    def _text_response(mocker: MockerFixture, content: str) -> provider.Reply:
        return make_completion_response(content)

    # A 200-word passage looping the same five words: low unique ratio, but
    # punctuated so it trips the diversity branch, not the run-on branch.
    LOOPING = "The lads sharted and then. " * 40
    # 160 distinct words with no sentence terminator: trips the run-on branch
    # while keeping unique ratio high, isolating that branch.
    RUN_ON = " ".join(f"word{i}" for i in range(160))
    # A vivid, well-formed long scene — the false-positive case the guard
    # must NOT flag: 150+ words, varied vocabulary, real sentence breaks.
    CLEAN_SCENE = (
        "The lab lights flickered violet as the Stinky Lads kicked open the "
        "double doors. Dan vaulted the front bench, scattering beakers like "
        "bowling pins. A green fog of methane rolled across the floor while "
        "Professor Blenkinsop dove behind the fume hood. Somewhere a Bunsen "
        "burner roared to life, casting jagged shadows up the periodic table. "
        "Mikey skidded across a puddle of spilled acid, cackling like a hyena. "
        "The fire alarm shrieked, drowned out by the lads chanting their "
        "dinner-hall anthem. Test tubes shattered against the whiteboard in a "
        "spray of glittering shards. Outside, seagulls scattered from the "
        "rooftop as smoke curled through a cracked window. Dan grabbed the "
        "intercom and bellowed a garbled war cry. The caretaker sprinted down "
        "the corridor, mop raised like a halberd. Glass crunched underfoot as "
        "they regrouped near the storeroom. Blenkinsop emerged, soot-streaked "
        "and furious, shaking a singed register at the smoke-filled ceiling."
    )

    @pytest.mark.parametrize("reply", [LOOPING, RUN_ON])
    def test_is_degraded_reply_detects_collapse(self, reply: str) -> None:
        """GIVEN run-on or looping prose WHEN checked THEN it is degraded."""
        assert _is_degraded_reply(reply) is True

    @pytest.mark.parametrize(
        "reply",
        [
            CLEAN_SCENE,
            # Short replies are never judged — too small a sample.
            "and then and then and then the lads ran and ran and ran off.",
            "Stinky Dan kicked the door wide, unleashing a volley of guff.",
            "",
        ],
    )
    def test_is_degraded_reply_false_for_clean_or_short(self, reply: str) -> None:
        """GIVEN a clean scene or a short reply WHEN checked THEN not degraded."""
        assert _is_degraded_reply(reply) is False

    def test_is_degraded_reply_ellipsis_prose_not_flagged(self) -> None:
        """Long, varied prose broken mainly with the Unicode ellipsis (…) must
        not read as a pathological run-on: … is a sentence/clause terminator.

        Before the fix only ASCII . ! ? were counted, so an ellipsis-heavy
        scene with a single period had words_per_sentence far over the run-on
        threshold and was wrongly stripped + retried.
        """
        words = [f"word{i}" for i in range(160)]  # 160 distinct words, high diversity
        reply = "… ".join(words) + "."
        assert len(reply.split()) >= 150  # long enough to be judged
        assert _is_degraded_reply(reply) is False

    def test_strip_degraded_removes_assistant_collapses(self) -> None:
        """GIVEN a thread with a collapsed turn WHEN stripped THEN only the
        model's degraded turn drops; user and clean turns are kept."""
        history = [
            {"role": "user", "content": "what happened in the science lab"},
            {"role": "assistant", "content": self.LOOPING},
            {"role": "user", "content": "and after that"},
            {"role": "assistant", "content": "The lads stormed the dorm next."},
        ]
        assert _strip_degraded(history) == [
            {"role": "user", "content": "what happened in the science lab"},
            {"role": "user", "content": "and after that"},
            {"role": "assistant", "content": "The lads stormed the dorm next."},
        ]

    def test_strip_degraded_handles_none_and_empty(self) -> None:
        """GIVEN no history WHEN stripped THEN it is returned unchanged."""
        assert _strip_degraded(None) is None
        assert _strip_degraded([]) == []

    def test_strip_degraded_keeps_clean_thread_intact(self) -> None:
        """GIVEN a thread with no collapses WHEN stripped THEN nothing drops."""
        history = [
            {"role": "user", "content": "set the scene"},
            {"role": "assistant", "content": self.CLEAN_SCENE},
        ]
        assert _strip_degraded(history) == history

    def test_trim_history_window_keeps_last_n(self) -> None:
        """GIVEN history longer than the window WHEN trimmed THEN only the
        most recent entries survive."""
        history = [{"role": "user", "content": str(i)} for i in range(15)]
        trimmed = _trim_history_window(history, 10)
        assert trimmed == history[-10:]
        assert len(trimmed) == 10

    @pytest.mark.parametrize("max_messages", [10, 0, -1])
    def test_trim_history_window_noop_cases(self, max_messages: int) -> None:
        """GIVEN history within the window, None, empty, or a non-positive
        cap WHEN trimmed THEN it is returned unchanged."""
        short = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
        assert _trim_history_window(short, max_messages) == short
        assert _trim_history_window(None, max_messages) is None
        assert _trim_history_window([], max_messages) == []

    def test_depoison_verse_history_strips_both_and_windows(self) -> None:
        """GIVEN a thread with a denial, a collapse, clean turns, and more
        than the window WHEN de-poisoned THEN both poisoned assistant turns
        are gone and the result is capped at the verse window."""
        history = [
            {"role": "user", "content": "what happened at alton towers"},
            {"role": "assistant", "content": "That never happened, pure fiction not in the canon."},
            {"role": "user", "content": "and the science lab"},
            {"role": "assistant", "content": self.LOOPING},
            *(
                msg
                for i in range(12)
                for msg in (
                    {"role": "user", "content": f"then what {i}"},
                    {"role": "assistant", "content": f"The lads charged onward, scene {i}."},
                )
            ),
        ]
        result = _depoison_verse_history(history)
        assert result is not None
        # Windowed to the verse cap.
        assert len(result) == 10
        # Neither poisoned turn survives.
        assert all(_is_verse_denial(m["content"]) is False for m in result)
        assert all(_is_degraded_reply(m["content"]) is False for m in result)

    def test_retries_and_recovers_when_verse_reply_collapses(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """GIVEN a verse reply collapses once WHEN it recovers on retry THEN
        assistant_completion returns the clean scene, not the gibberish."""
        from llm.service import PROFILE_VERSE

        story = "The lab erupted as the Stinky Lads stormed in with guff-grenades."
        responses = [
            self._text_response(mocker, self.LOOPING),
            self._text_response(mocker, story),
        ]
        seen_messages: list[list] = []

        def fake_completion(**kwargs: object) -> MagicMock:
            seen_messages.append(list(kwargs.get("messages", [])))  # type: ignore[arg-type]
            return responses[len(seen_messages) - 1]

        mocker.patch("llm.service.provider.chat", side_effect=fake_completion)

        result = service.assistant_completion(
            prompt="what happened in the science lab",
            nick="fc42",
            channel="#afternet",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="vibebot",
            route_profile=PROFILE_VERSE,
        )

        assert result.content == story
        assert result.error is None
        assert len(seen_messages) == 2
        # The retry call carries the corrective nudge as a user message.
        assert any(
            m.get("role") == "user" and "Rewrite it cleanly" in str(m.get("content", ""))
            for m in seen_messages[1]
        )

    def test_chat_profile_retries_and_recovers_when_reply_collapses(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """GIVEN a chat reply (the @ask fallback) collapses once WHEN it
        recovers on retry THEN the clean answer is returned, not the
        gibberish — the guard is not verse-only."""
        from llm.service import PROFILE_CHAT

        answer = "The meeting covered the budget, the roadmap, and the hiring plan."
        responses = [
            self._text_response(mocker, self.LOOPING),
            self._text_response(mocker, answer),
        ]
        seen: list[list] = []

        def fake_completion(**kwargs: object) -> MagicMock:
            seen.append(list(kwargs.get("messages", [])))  # type: ignore[arg-type]
            return responses[len(seen) - 1]

        mocker.patch("llm.service.provider.chat", side_effect=fake_completion)

        result = service.assistant_completion(
            prompt="summarize the meeting",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="vibebot",
            route_profile=PROFILE_CHAT,
        )

        assert result.content == answer
        assert result.error is None
        assert len(seen) == 2
        # The retry call carries the corrective nudge as a user message.
        assert any(
            m.get("role") == "user" and "Rewrite it cleanly" in str(m.get("content", ""))
            for m in seen[1]
        )

    def test_returns_best_effort_when_verse_keeps_collapsing(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """GIVEN a verse reply collapses on every step WHEN the retry budget
        is spent THEN the last attempt is delivered (not an error)."""
        from llm.service import PROFILE_VERSE

        mocker.patch(
            "llm.service.provider.chat",
            side_effect=lambda **_kw: self._text_response(mocker, self.LOOPING),
        )

        result = service.assistant_completion(
            prompt="what happened in the dorm",
            nick="fc42",
            channel="#afternet",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="vibebot",
            route_profile=PROFILE_VERSE,
        )

        assert result.content
        assert result.error is None


class TestRepeatReplyGuard:
    """Tests for the cross-turn self-repetition guard.

    Distinct from the quality-collapse guard (which needs a 150+ word
    passage): the failure here is a SHORT reply the bot converges on and
    parrots across turns and days ("Riding a flaming cheese comet…"),
    re-seeded every turn by its own stored conversation history.
    _replies_repetitive detects near-duplicate reply pairs;
    _strip_repeated_replies drops the whole duplicate cluster from history
    (a lone survivor would just re-seed the schtick); the loop nudges and
    retries once when a fresh reply parrots a prior one.
    """

    @pytest.fixture
    def service(self, make_service) -> LLMService:  # type: ignore[no-untyped-def]
        svc, _plugin = make_service(assistantModel="gpt-4")
        return svc

    @staticmethod
    def _text_response(mocker: MockerFixture, content: str) -> provider.Reply:
        return make_completion_response(content)

    # Real production pair: the bot's greeting reply on consecutive days —
    # different verbs and endings, same stuck schtick.
    COMET_A = (
        "Bro I'm surfing a flaming cheese comet through exploding retro "
        "game galaxies while 3D printing infinite Toronto wineries."
    )
    COMET_B = (
        "Riding a flaming cheese comet through exploding retro game "
        "galaxies while 3D-printing wine barrels on Ubuntu!"
    )
    # Shares a couple of words with the comets but is a different line —
    # must NOT be treated as a repeat.
    GREETING = "HI FROM THE COSMIC CHEESE VOID, RETRO GAMER!"
    FRESH = "Just holding down the channel, mate. What are you up to?"

    def test_replies_repetitive_detects_stuck_schtick(self) -> None:
        """GIVEN two near-duplicate replies WHEN compared THEN repetitive."""
        assert _replies_repetitive(self.COMET_A, self.COMET_B) is True

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            (COMET_B, GREETING),
            (COMET_A, FRESH),
            # Short but genuinely different replies stay clear of each other.
            ("Done.", "No chance."),
            ("", COMET_A),
        ],
    )
    def test_replies_repetitive_false_for_distinct(self, a: str, b: str) -> None:
        """GIVEN distinct replies WHEN compared THEN not repetitive."""
        assert _replies_repetitive(a, b) is False

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            # Four words, the line that went out nine times in ninety seconds
            # (#afternet, 2026-08-24 21:48) under the old five-word floor.
            ("fc42, you absolute wanker.", "Fc42, you absolute wanker."),
            # Three words, the report a whole separate guard was written to
            # catch because this one would not look at it.
            ("Image generation failed.", "Image generation failed."),
            ("Done.", "Done."),
        ],
    )
    def test_short_replies_are_judged_too(self, a: str, b: str) -> None:
        """Length is not a defence — a stuck record is a stuck record."""
        assert _replies_repetitive(a, b) is True

    def test_strip_repeated_replies_drops_whole_cluster(self) -> None:
        """GIVEN a thread with two near-duplicate assistant turns WHEN
        stripped THEN both drop (no survivor to re-seed) while user turns
        and distinct assistant turns are kept."""
        history = [
            {"role": "user", "content": "how's it going"},
            {"role": "assistant", "content": self.COMET_A},
            {"role": "user", "content": "say hi"},
            {"role": "assistant", "content": self.GREETING},
            {"role": "user", "content": "how's it going"},
            {"role": "assistant", "content": self.COMET_B},
        ]
        assert _strip_repeated_replies(history) == [
            {"role": "user", "content": "how's it going"},
            {"role": "user", "content": "say hi"},
            {"role": "assistant", "content": self.GREETING},
            {"role": "user", "content": "how's it going"},
        ]

    def test_strip_repeated_replies_keeps_clean_thread_intact(self) -> None:
        """GIVEN a thread with no repeats WHEN stripped THEN nothing drops."""
        history = [
            {"role": "user", "content": "how's it going"},
            {"role": "assistant", "content": self.COMET_A},
            {"role": "user", "content": "what's new"},
            {"role": "assistant", "content": self.FRESH},
        ]
        assert _strip_repeated_replies(history) == history

    def test_strip_repeated_replies_handles_none_and_empty(self) -> None:
        """GIVEN no history WHEN stripped THEN it is returned unchanged."""
        assert _strip_repeated_replies(None) is None
        assert _strip_repeated_replies([]) == []

    def test_chat_retries_and_recovers_when_reply_parrots_history(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """GIVEN stored history containing a past reply WHEN the fresh reply
        parrots it THEN the loop nudges and retries, and the fresh answer is
        returned instead of the repeat."""
        responses = [
            self._text_response(mocker, self.COMET_A),
            self._text_response(mocker, self.FRESH),
        ]
        seen: list[list] = []

        def fake_completion(**kwargs: object) -> MagicMock:
            seen.append(list(kwargs.get("messages", [])))  # type: ignore[arg-type]
            return responses[len(seen) - 1]

        mocker.patch("llm.service.provider.chat", side_effect=fake_completion)

        result = service.assistant_completion(
            prompt="how's it going",
            nick="rdrake",
            channel="#afternet",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="vibebot",
            history=[
                {"role": "user", "content": "how's it going"},
                {"role": "assistant", "content": self.COMET_B},
            ],
        )

        assert result.content == self.FRESH
        assert result.error is None
        assert len(seen) == 2
        # The retry call carries the corrective nudge as a user message.
        assert any(
            m.get("role") == "user" and str(m.get("content", "")) == _REPEAT_RETRY_NUDGE
            for m in seen[1]
        )

    def test_chat_retries_reply_matching_stripped_duplicate_cluster(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """GIVEN persisted duplicate replies WHEN the model repeats them THEN
        they stay out of the prompt but remain anchors for the retry guard."""
        responses = [
            self._text_response(mocker, self.COMET_A),
            self._text_response(mocker, self.FRESH),
        ]
        seen: list[list] = []

        def fake_completion(**kwargs: object) -> MagicMock:
            seen.append(list(kwargs.get("messages", [])))  # type: ignore[arg-type]
            return responses[len(seen) - 1]

        mocker.patch("llm.service.provider.chat", side_effect=fake_completion)

        result = service.assistant_completion(
            prompt="how's it going",
            nick="rdrake",
            channel="#afternet",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="vibebot",
            history=[
                {"role": "user", "content": "how's it going"},
                {"role": "assistant", "content": self.COMET_A},
                {"role": "user", "content": "same question tomorrow"},
                {"role": "assistant", "content": self.COMET_B},
            ],
        )

        assert result.content == self.FRESH
        assert result.error is None
        assert len(seen) == 2
        assert all(m.get("content") not in {self.COMET_A, self.COMET_B} for m in seen[0])
        assert any(
            m.get("role") == "user" and str(m.get("content", "")) == _REPEAT_RETRY_NUDGE
            for m in seen[1]
        )

    def test_degraded_history_reply_does_not_anchor_repetition(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """GIVEN a degraded stored reply WHEN a clean reply resembles it THEN
        the reply is delivered without a repetition retry."""
        degraded = "The lads sharted and then. " * 40
        candidate = "The lads sharted and then."
        calls = 0
        assert _is_degraded_reply(degraded) is True
        assert _is_degraded_reply(candidate) is False
        assert _replies_repetitive(candidate, degraded) is True

        def fake_completion(**_kwargs: object) -> MagicMock:
            nonlocal calls
            calls += 1
            return self._text_response(mocker, candidate)

        mocker.patch("llm.service.provider.chat", side_effect=fake_completion)

        result = service.assistant_completion(
            prompt="continue with something concise",
            nick="rdrake",
            channel="#afternet",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="vibebot",
            history=[{"role": "assistant", "content": degraded}],
        )

        assert result.content == candidate
        assert result.error is None
        assert calls == 1

    def test_verse_denial_history_reply_does_not_anchor_repetition(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """GIVEN a stored verse denial WHEN a valid reply resembles it THEN
        the reply is delivered without a repetition retry."""
        from llm.service import PROFILE_VERSE

        denial = "That never happened at all, pure fiction not in the canon."
        candidate = "In the canon, that fiction became spectacle at all."
        calls = 0
        assert _is_verse_denial(denial) is True
        assert _is_verse_denial(candidate) is False
        assert _replies_repetitive(candidate, denial) is True

        def fake_completion(**_kwargs: object) -> MagicMock:
            nonlocal calls
            calls += 1
            return self._text_response(mocker, candidate)

        mocker.patch("llm.service.provider.chat", side_effect=fake_completion)

        result = service.assistant_completion(
            prompt="continue the scene",
            nick="rdrake",
            channel="#afternet",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="vibebot",
            route_profile=PROFILE_VERSE,
            history=[{"role": "assistant", "content": denial}],
        )

        assert result.content == candidate
        assert result.error is None
        assert calls == 1

    def test_verse_channel_history_reply_does_not_anchor_repetition(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """GIVEN a matching reply only in verse channel history WHEN the
        model answers THEN channel chatter does not trigger a retry."""
        from llm.service import PROFILE_VERSE

        calls = 0
        assert _replies_repetitive(self.COMET_A, self.COMET_B) is True

        def fake_completion(**_kwargs: object) -> MagicMock:
            nonlocal calls
            calls += 1
            return self._text_response(mocker, self.COMET_A)

        mocker.patch("llm.service.provider.chat", side_effect=fake_completion)

        result = service.assistant_completion(
            prompt="continue the scene",
            nick="rdrake",
            channel="#afternet",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="vibebot",
            route_profile=PROFILE_VERSE,
            channel_history=[{"role": "assistant", "content": self.COMET_B}],
        )

        assert result.content == self.COMET_A
        assert result.error is None
        assert calls == 1

    def test_chat_delivers_best_effort_when_repeat_persists(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """GIVEN the model keeps parroting after the retry budget WHEN the
        loop finishes THEN the best-effort reply is still delivered."""
        mocker.patch(
            "llm.service.provider.chat",
            side_effect=lambda **_kw: self._text_response(mocker, self.COMET_A),
        )

        result = service.assistant_completion(
            prompt="how's it going",
            nick="rdrake",
            channel="#afternet",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="vibebot",
            history=[
                {"role": "user", "content": "how's it going"},
                {"role": "assistant", "content": self.COMET_B},
            ],
        )

        assert result.content == self.COMET_A
        assert result.error is None


# =========================================================================
# Plugin-level command tests
# =========================================================================


class TestReminderMetaHelpers:
    """Tests for plugin reminder helper methods used by meta."""

    @pytest.fixture
    def plugin(self, mocker: MockerFixture, mock_irc: MagicMock):  # type: ignore[no-untyped-def]
        import threading

        plugin_init_patches(mocker)
        plugin = LLM(mock_irc)
        plugin.registryValue = mocker.Mock(side_effect=make_registry_side_effect({}))
        plugin.llm_service = mocker.MagicMock()
        plugin.llm_service.sanitize_output.side_effect = lambda s: s
        plugin.db.get_user_timezone.return_value = None
        plugin.db.get_clock_offset.return_value = None
        plugin.db.load_pending_reminders.return_value = []
        plugin._MetaSynchronized_rlock = threading.RLock()
        return plugin

    def test_remind_set_for_assistant_success(
        self, plugin, mocker: MockerFixture, mock_irc: MagicMock
    ) -> None:
        """GIVEN valid reminder text WHEN _remind_set_for_assistant THEN returns confirmation."""
        from llm.service import ReminderParseResult

        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="schedule",
            seconds=3600,
            message="check the build",
            confirmation="I'll remind you in 1 hour",
        )
        mocker.patch("llm.plugin.schedule.addEvent")

        msg = mocker.MagicMock()
        msg.args = ["#test"]

        result = plugin._remind_set_for_assistant(
            mock_irc,
            msg,
            Caller.from_row("testuser", None, msg.args[0]),
            "check the build in 1 hour",
        )

        assert result.ok is True
        assert "remind" in result.message.lower() or "hour" in result.message.lower()
        assert plugin.db.save_reminder.called

    def test_remind_set_for_assistant_with_note(
        self, plugin, mocker: MockerFixture, mock_irc: MagicMock
    ) -> None:
        """GIVEN reminder with note WHEN _remind_set_for_assistant THEN includes note."""
        from llm.service import ReminderParseResult

        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="schedule",
            seconds=3600,
            message="deploy",
            confirmation="I'll remind you in 1 hour",
            note="assuming Eastern time",
        )
        mocker.patch("llm.plugin.schedule.addEvent")

        msg = mocker.MagicMock()
        msg.args = ["#test"]

        result = plugin._remind_set_for_assistant(
            mock_irc, msg, Caller.from_row("testuser", None, msg.args[0]), "deploy in 1 hour"
        )

        assert "Eastern" in result.message

    def test_remind_set_for_assistant_parse_failure(
        self, plugin, mocker: MockerFixture, mock_irc: MagicMock
    ) -> None:
        """GIVEN unparseable reminder WHEN _remind_set_for_assistant THEN returns error."""
        from llm.service import ReminderParseResult

        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="schedule",
            seconds=None,
        )

        msg = mocker.MagicMock()
        msg.args = ["#test"]

        result = plugin._remind_set_for_assistant(
            mock_irc, msg, Caller.from_row("testuser", None, msg.args[0]), "maybe sometime"
        )

        assert result.ok is False
        assert "could not" in result.message.lower()

    def test_remind_set_for_assistant_too_short(
        self, plugin, mocker: MockerFixture, mock_irc: MagicMock
    ) -> None:
        """GIVEN reminder < 10 seconds WHEN _remind_set_for_assistant THEN returns error."""
        from llm.service import ReminderParseResult

        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="schedule",
            seconds=5,
            message="now",
            confirmation="OK",
        )

        msg = mocker.MagicMock()
        msg.args = ["#test"]

        result = plugin._remind_set_for_assistant(
            mock_irc, msg, Caller.from_row("testuser", None, msg.args[0]), "remind me now"
        )

        assert result.ok is False
        assert "10 second" in result.message.lower() or "at least" in result.message.lower()

    def test_remind_set_for_assistant_too_long(
        self, plugin, mocker: MockerFixture, mock_irc: MagicMock
    ) -> None:
        """GIVEN reminder > 7 days WHEN _remind_set_for_assistant THEN returns error."""
        from llm.service import ReminderParseResult

        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="schedule",
            seconds=700000,
            message="later",
            confirmation="OK",
        )

        msg = mocker.MagicMock()
        msg.args = ["#test"]

        result = plugin._remind_set_for_assistant(
            mock_irc, msg, Caller.from_row("testuser", None, msg.args[0]), "remind me in 2 weeks"
        )

        assert result.ok is False
        assert "7 day" in result.message.lower()

    def test_remind_set_for_assistant_clarify(
        self, plugin, mocker: MockerFixture, mock_irc: MagicMock
    ) -> None:
        """GIVEN clarify action WHEN _remind_set_for_assistant THEN nothing is set:
        the tool fails (so the chat reply carrying the question isn't dropped as
        a redundant ack), passes the question on, and reacts neither ⏰ nor ❌.
        On 2026-10-19 a question read as success showed ⏰ and the user never
        saw it."""
        from llm.service import ReminderParseResult

        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="clarify",
            confirmation="When exactly should I remind you?",
        )
        react = mocker.patch.object(plugin, "_react")

        msg = mocker.MagicMock()
        msg.args = ["#test"]

        result = plugin._remind_set_for_assistant(
            mock_irc, msg, Caller.from_row("testuser", None, msg.args[0]), "remind me"
        )

        assert result.ok is False
        assert "When exactly should I remind you?" in result.message
        react.assert_not_called()
        plugin.db.save_reminder.assert_not_called()

    def test_remind_command_clarify_asks_without_reacting(
        self, plugin, mocker: MockerFixture, mock_irc: MagicMock
    ) -> None:
        """GIVEN clarify action WHEN @remind runs THEN the question is the reply,
        with no ⏰ (nothing was set) and no ❌ (nothing failed)."""
        from llm.service import ReminderParseResult

        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="clarify",
            confirmation="When exactly should I remind you?",
        )
        react = mocker.patch.object(plugin, "_react")
        reply = mocker.patch.object(plugin, "_safe_reply")

        msg = mocker.MagicMock()
        msg.args = ["#test"]

        plugin._remind_set(
            mock_irc, msg, Caller.from_row("testuser", None, msg.args[0]), "remind me"
        )

        react.assert_not_called()
        reply.assert_called_once_with(
            mock_irc, "When exactly should I remind you?", prefixNick=True
        )

    def test_remind_delete_for_assistant_success(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN valid reminder ID WHEN _remind_delete_for_assistant THEN deletes."""
        event_name = "llm_remind_abc123def456"
        row = make_reminder_row(
            event_name=event_name,
            nick="testuser",
            channel="#test",
            message="check build",
        )
        plugin.db.load_pending_reminders.return_value = [row]
        plugin.db.get_reminder.return_value = row

        result = plugin._remind_delete_for_assistant(
            Identity(raw_nick="testuser", account=None), "abc123def456"
        )

        assert result.ok is True
        assert "delete" in result.message.lower() or "cancel" in result.message.lower()
        plugin.db.delete_reminder.assert_called_once_with(event_name)

    def test_remind_delete_for_assistant_not_found(self, plugin) -> None:
        """GIVEN unknown reminder ID WHEN _remind_delete_for_assistant THEN error."""

        result = plugin._remind_delete_for_assistant(
            Identity(raw_nick="testuser", account=None), "nonexistent"
        )

        assert result.ok is False
        assert "not found" in result.message.lower()


class TestDrawForMeta:
    """Tests for _draw_for_assistant helper used by meta generate_image tool."""

    @pytest.fixture
    def plugin(self, mocker: MockerFixture, mock_irc: MagicMock):  # type: ignore[no-untyped-def]
        import threading

        plugin_init_patches(mocker)
        plugin = LLM(mock_irc)
        plugin.registryValue = mocker.Mock(side_effect=make_registry_side_effect())
        plugin.llm_service = mocker.MagicMock()
        plugin.db = mocker.MagicMock()
        plugin._MetaSynchronized_rlock = threading.RLock()
        return plugin

    def test_draw_for_assistant_logs_its_own_usage_row(
        self, plugin, mocker: MockerFixture, mock_irc: MagicMock
    ) -> None:
        """_draw_for_assistant books image spend under the IMAGE model.

        This is the documented exception to "leaf tool handlers do not log
        independently". A usage row names exactly one model, and this leaf
        spends on a different one from the turn that invoked it, so folding its
        cost into the caller's row files image spend under whatever chat model
        answered. Since 2026-04-11 every path to an image runs through here, so
        that was every image the bot drew.
        """
        from llm.service import ImageResult

        plugin.llm_service.image_generation.return_value = ImageResult(
            content="https://img.example/cat.png",
            model="xai/grok-imagine-image",
            prompt_tokens=10,
            completion_tokens=0,
            cost=0.02,
        )

        msg = mocker.MagicMock()
        msg.prefix = "user!ident@host"
        msg.args = ["#test"]

        result = plugin._draw_for_assistant(mock_irc, msg, "a cat")

        assert result.ok is True
        assert result.message == "https://img.example/cat.png"
        args, kwargs = plugin.db.log_usage.call_args
        assert args[2] == "draw:image"
        assert args[3] == "xai/grok-imagine-image"
        assert args[6] == 0.02
        assert kwargs["status"] == "success"

    def test_draw_for_assistant_forwards_the_aspect(
        self, plugin, mocker: MockerFixture, mock_irc: MagicMock
    ) -> None:
        """GIVEN an aspect WHEN drawing for the tool THEN image_generation receives it."""
        from llm.service import ImageResult

        plugin.llm_service.image_generation.return_value = ImageResult(
            content="https://img.example/poster.png", model="xai/grok-imagine-image"
        )
        msg = mocker.MagicMock()
        msg.prefix = "user!ident@host"
        msg.args = ["#test"]

        plugin._draw_for_assistant(mock_irc, msg, "a poster", aspect="portrait")

        assert plugin.llm_service.image_generation.call_args.kwargs["aspect"] == "portrait"

    def test_draw_for_assistant_returns_no_usage_to_the_caller(
        self, plugin, mocker: MockerFixture, mock_irc: MagicMock
    ) -> None:
        """Having logged its own row, it must not also report cost upward.

        The executor accumulates whatever a leaf returns and the wrapper logs
        that. Returning cost here as well would put the same spend in two rows.
        """
        from llm.service import ImageResult

        plugin.llm_service.image_generation.return_value = ImageResult(
            content="https://img.example/cat.png",
            model="xai/grok-imagine-image",
            prompt_tokens=10,
            completion_tokens=0,
            cost=0.02,
        )

        msg = mocker.MagicMock()
        msg.prefix = "user!ident@host"
        msg.args = ["#test"]

        result = plugin._draw_for_assistant(mock_irc, msg, "a cat")

        assert not hasattr(result, "cost")
        assert not hasattr(result, "prompt_tokens")

    def test_draw_for_assistant_skips_the_row_when_nothing_was_spent(
        self, plugin, mocker: MockerFixture, mock_irc: MagicMock
    ) -> None:
        """A prompt rejected before any provider call is not a purchase.

        validate_prompt failures and missing API keys never reach the provider;
        a zero row would only dilute the per-image averages.
        """
        from llm.service import ImageResult

        plugin.llm_service.image_generation.return_value = ImageResult(
            content="Error: prompt too long",
            error="Error: prompt too long",
        )

        msg = mocker.MagicMock()
        msg.prefix = "user!ident@host"
        msg.args = ["#test"]

        plugin._draw_for_assistant(mock_irc, msg, "a cat")

        plugin.db.log_usage.assert_not_called()

    def test_draw_for_assistant_books_a_billed_refusal(
        self, plugin, mocker: MockerFixture, mock_irc: MagicMock
    ) -> None:
        """A refused generation the provider charged for still gets a row."""
        from llm.service import ImageResult

        plugin.llm_service.image_generation.return_value = ImageResult(
            content="Error: No image generated. The prompt was blocked by content safety filters.",
            model="xai/grok-imagine-image",
            cost=0.02,
            error="Error: No image generated. The prompt was blocked by content safety filters.",
        )

        msg = mocker.MagicMock()
        msg.prefix = "user!ident@host"
        msg.args = ["#test"]

        plugin._draw_for_assistant(mock_irc, msg, "a cat")

        args, kwargs = plugin.db.log_usage.call_args
        assert args[6] == 0.02
        assert kwargs["status"] == "content_blocked"

    def test_recovered_draw_writes_one_row_per_provider_call(
        self, plugin, mocker: MockerFixture, mock_irc: MagicMock
    ) -> None:
        """A refusal the rewrite recovered from is a row of its own.

        Booking the turn as a single success hides the refusal AND the prompt
        that caused it, which is the only text that can answer whether the chat
        model embellishes its own tool argument into a block.
        """
        from llm.service import BlockedAttempt, ImageResult

        plugin.llm_service.image_generation.return_value = ImageResult(
            content="https://img.example/cat.png",
            model="xai/grok-imagine-image",
            cost=0.0403,
            rewritten_prompt="a cat",
            blocked_attempts=(
                BlockedAttempt("a cat, dramatically on fire", "content moderation", 0.02),
            ),
        )

        msg = mocker.MagicMock()
        msg.prefix = "user!ident@host"
        msg.args = ["#test"]

        plugin._draw_for_assistant(mock_irc, msg, "a cat, dramatically on fire")

        assert plugin.db.log_usage.call_count == 2
        blocked_args, blocked_kwargs = plugin.db.log_usage.call_args_list[0]
        assert blocked_kwargs["status"] == "content_blocked"
        assert blocked_kwargs["prompt"] == "a cat, dramatically on fire"
        assert "content moderation" in blocked_kwargs["error_detail"]
        assert blocked_args[6] == 0.02
        assert plugin.db.log_usage.call_args_list[1][1]["status"] == "success"

    def test_rows_of_a_recovered_draw_sum_to_what_the_call_spent(
        self, plugin, mocker: MockerFixture, mock_irc: MagicMock
    ) -> None:
        """Splitting the bill must not mint money or lose it.

        ``ImageResult.cost`` is every attempt plus the rewriter; the blocked
        rows take their own share and the delivered row takes the remainder.
        """
        from llm.service import BlockedAttempt, ImageResult

        plugin.llm_service.image_generation.return_value = ImageResult(
            content="https://img.example/cat.png",
            model="xai/grok-imagine-image",
            cost=0.0403,
            blocked_attempts=(BlockedAttempt("a cat on fire", "content moderation", 0.02),),
        )

        msg = mocker.MagicMock()
        msg.prefix = "user!ident@host"
        msg.args = ["#test"]

        plugin._draw_for_assistant(mock_irc, msg, "a cat on fire")

        booked = sum(call[0][6] for call in plugin.db.log_usage.call_args_list)
        assert booked == pytest.approx(0.0403)

    def test_free_refusal_still_gets_its_row(
        self, plugin, mocker: MockerFixture, mock_irc: MagicMock
    ) -> None:
        """A draw blocked with no picture returned can charge nothing for it.

        The "nothing was spent, skip the row" rule is about calls that never
        reached a provider. This one reached one and came back refused; the
        prompt is the evidence even when the money is zero.
        """
        from llm.service import BlockedAttempt, ImageResult

        plugin.llm_service.image_generation.return_value = ImageResult(
            content="https://img.example/cat.png",
            model="xai/grok-imagine-image",
            cost=0.0,
            blocked_attempts=(BlockedAttempt("a cat on fire", "empty response", 0.0),),
        )

        msg = mocker.MagicMock()
        msg.prefix = "user!ident@host"
        msg.args = ["#test"]

        plugin._draw_for_assistant(mock_irc, msg, "a cat on fire")

        assert plugin.db.log_usage.call_count == 2
        assert plugin.db.log_usage.call_args_list[0][1]["status"] == "content_blocked"

    def test_a_blocked_row_that_fails_to_write_does_not_sink_the_image(
        self, plugin, mocker: MockerFixture, mock_irc: MagicMock
    ) -> None:
        """Same rule as the main row: accounting must not cost the user a picture."""
        from llm.service import BlockedAttempt, ImageResult

        plugin.llm_service.image_generation.return_value = ImageResult(
            content="https://img.example/cat.png",
            model="xai/grok-imagine-image",
            cost=0.0403,
            blocked_attempts=(BlockedAttempt("a cat on fire", "content moderation", 0.02),),
        )
        plugin.db.log_usage.side_effect = RuntimeError("disk full")

        msg = mocker.MagicMock()
        msg.prefix = "user!ident@host"
        msg.args = ["#test"]

        result = plugin._draw_for_assistant(mock_irc, msg, "a cat on fire")

        assert result.ok is True
        assert result.message == "https://img.example/cat.png"

    def test_usage_logging_failure_does_not_sink_the_image(
        self, plugin, mocker: MockerFixture, mock_irc: MagicMock
    ) -> None:
        """Accounting is bookkeeping; the user is waiting for a picture."""
        from llm.service import ImageResult

        plugin.llm_service.image_generation.return_value = ImageResult(
            content="https://img.example/cat.png",
            model="xai/grok-imagine-image",
            cost=0.02,
        )
        plugin.db.log_usage.side_effect = RuntimeError("disk full")

        msg = mocker.MagicMock()
        msg.prefix = "user!ident@host"
        msg.args = ["#test"]

        result = plugin._draw_for_assistant(mock_irc, msg, "a cat")

        assert result.ok is True
        assert result.message == "https://img.example/cat.png"

    def test_draw_for_assistant_returns_content(
        self, plugin, mocker: MockerFixture, mock_irc: MagicMock
    ) -> None:
        """_draw_for_assistant returns the image result content string."""
        from llm.service import ImageResult

        plugin.llm_service.image_generation.return_value = ImageResult(
            content="https://img.example/sunset.png",
            model="dall-e-3",
            prompt_tokens=5,
            completion_tokens=0,
            cost=0.02,
        )

        msg = mocker.MagicMock()
        msg.prefix = "user!ident@host"
        msg.args = ["#test"]

        result = plugin._draw_for_assistant(mock_irc, msg, "a sunset")

        assert result.ok is True
        assert result.message == "https://img.example/sunset.png"
        plugin.llm_service.image_generation.assert_called_once_with(
            "a sunset", irc=mock_irc, msg=msg, aspect=None
        )


class TestCodeForAssistant:
    """Tests for _code_for_assistant helper used by meta generate_code tool."""

    @pytest.fixture
    def plugin(self, mocker: MockerFixture, mock_irc: MagicMock):  # type: ignore[no-untyped-def]
        import threading

        plugin_init_patches(mocker)
        plugin = LLM(mock_irc)
        plugin.registryValue = mocker.Mock(
            side_effect=make_registry_side_effect({"codeSystemPrompt": "You are a coder."})
        )
        plugin.llm_service = mocker.MagicMock()
        plugin.db = mocker.MagicMock()
        plugin._MetaSynchronized_rlock = threading.RLock()
        return plugin

    def test_code_for_assistant_returns_url(self, plugin) -> None:
        """_code_for_assistant saves code to HTTP and returns URL."""
        from llm.service import CompletionResult

        plugin.llm_service.completion.return_value = CompletionResult(
            content="print('hello')",
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )
        plugin.llm_service.save_code_to_http.return_value = "https://example.com/code/abc123"

        result = plugin._code_for_assistant("write hello world", "#test")

        import json

        data = json.loads(result.content)
        assert data["url"] == "https://example.com/code/abc123"
        plugin.llm_service.save_code_to_http.assert_called_once_with(
            "print('hello')", title="write hello world"
        )

    def test_code_for_assistant_includes_code_in_result(self, plugin) -> None:
        """_code_for_assistant includes code content for context."""
        from llm.service import CompletionResult

        plugin.llm_service.completion.return_value = CompletionResult(
            content="def foo(): pass",
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )
        plugin.llm_service.save_code_to_http.return_value = "https://example.com/code/xyz"

        result = plugin._code_for_assistant("write a function", "#test")

        import json

        data = json.loads(result.content)
        assert data["code"] == "def foo(): pass"
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 5
        assert result.cost == 0.001

    def test_code_for_assistant_handles_error(self, plugin) -> None:
        """_code_for_assistant returns sanitized error ToolResult on failure."""
        plugin.llm_service.completion.side_effect = RuntimeError("API down")

        result = plugin._code_for_assistant("write something", "#test")

        import json

        data = json.loads(result.content)
        assert "error" in data
        # Internal exception text must NOT leak into the tool payload
        assert "API down" not in data["error"]
        assert data["error"] == "Code generation failed."

    def test_code_for_assistant_handles_completion_error(self, plugin) -> None:
        """_code_for_assistant returns error ToolResult when completion has error."""
        from llm.service import CompletionResult

        plugin.llm_service.completion.return_value = CompletionResult(
            content="",
            error="Content blocked",
        )

        result = plugin._code_for_assistant("bad prompt", "#test")

        import json

        data = json.loads(result.content)
        assert data["error"] == "Content blocked"


class TestInvalidCommandRouting:
    """Tests for invalidCommand routing directly through chat profile."""

    @pytest.fixture
    def plugin(self, mocker: MockerFixture, mock_irc: MagicMock):  # type: ignore[no-untyped-def]
        plugin_init_patches(mocker)
        plugin = LLM(mock_irc)
        plugin.registryValue = mocker.Mock(side_effect=make_registry_side_effect({}))
        plugin.llm_service = mocker.MagicMock()
        plugin.db = mocker.MagicMock()
        return plugin

    def test_invalid_command_routes_through_ask_impl(
        self, plugin, mocker: MockerFixture, mock_irc: MagicMock
    ) -> None:
        """GIVEN unknown command WHEN invalidCommand THEN routes to _ask_impl with chat profile."""
        msg = mocker.MagicMock()
        msg.prefix = "user!ident@host"
        msg.nick = "testuser"
        msg.args = ["#test"]

        plugin._ask_impl = mocker.MagicMock()
        plugin._run_preflight = mocker.MagicMock(
            return_value=PreflightResult(False, Caller(Identity("testuser", None), "#test"))
        )

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin._is_old_message = mocker.MagicMock(return_value=False)

        plugin.invalidCommand(mock_irc, msg, ["what", "is", "python"])

        plugin._ask_impl.assert_called_once()
        call_kwargs = plugin._ask_impl.call_args
        assert call_kwargs[1]["entry_route"] == "invalid_command"

    def test_invalid_command_no_meta_dispatch(
        self, plugin, mocker: MockerFixture, mock_irc: MagicMock
    ) -> None:
        """GIVEN unknown command WHEN invalidCommand THEN does not call _run_meta."""
        msg = mocker.MagicMock()
        msg.prefix = "user!ident@host"
        msg.nick = "testuser"
        msg.args = ["#test"]

        plugin._ask_impl = mocker.MagicMock()
        plugin._run_meta = mocker.MagicMock()
        plugin._run_preflight = mocker.MagicMock(
            return_value=PreflightResult(False, Caller(Identity("testuser", None), "#test"))
        )

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin._is_old_message = mocker.MagicMock(return_value=False)

        plugin.invalidCommand(mock_irc, msg, ["hello", "there"])

        plugin._run_meta.assert_not_called()
        plugin.llm_service.assistant_completion.assert_not_called()

    def test_invalid_command_still_checks_capability(
        self, plugin, mocker: MockerFixture, mock_irc: MagicMock
    ) -> None:
        """GIVEN user without llm.ask capability WHEN invalidCommand THEN returns early."""
        msg = mocker.MagicMock()
        msg.prefix = "user!ident@host"
        msg.nick = "testuser"
        msg.args = ["#test"]

        plugin._ask_impl = mocker.MagicMock()

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=False)

        plugin.invalidCommand(mock_irc, msg, ["always", "respond", "in", "haiku"])

        plugin._ask_impl.assert_not_called()

    def test_addressed_dispatch_offloaded_to_daemon_thread(
        self, plugin, mocker: MockerFixture, mock_irc: MagicMock
    ) -> None:
        """Addressed dispatch runs on a daemon thread, not inline.

        Regression for typing lag: doPrivmsg/invalidCommand execute on the
        IRC driver thread, which only flushes the outbound queue after the
        callback returns. Running LLM generation inline pins the driver so
        the +typing indicator can't leave the socket until the reply is
        ready. The dispatch must be offloaded so the driver is freed to
        flush typing immediately.
        """
        msg = mocker.MagicMock()
        captured: dict = {}

        class _FakeThread:
            def __init__(self, target, name=None, daemon=None):  # type: ignore[no-untyped-def]
                captured["target"] = target
                captured["name"] = name
                captured["daemon"] = daemon
                captured["started"] = False

            def start(self) -> None:
                captured["started"] = True

        mocker.patch("llm.plugin.world.SupyThread", _FakeThread)
        plugin._dispatch_with_verse_routing = mocker.MagicMock()

        preflight = PreflightResult(False, Caller(Identity("u", None), "#c"))
        plugin._dispatch_addressed_async(
            mock_irc, msg, "hi there", preflight, entry_route="addressed"
        )

        # Created as a started daemon thread; dispatch has NOT run inline on
        # the calling (driver) thread.
        assert captured["daemon"] is True
        assert captured["started"] is True
        plugin._dispatch_with_verse_routing.assert_not_called()

        # Invoking the thread target performs the actual dispatch.
        captured["target"]()
        plugin._dispatch_with_verse_routing.assert_called_once()
        assert plugin._dispatch_with_verse_routing.call_args.kwargs["entry_route"] == "addressed"

    def test_addressed_dispatch_worker_bails_when_closing(
        self, plugin, mocker: MockerFixture, mock_irc: MagicMock
    ) -> None:
        """If shutdown began before the worker runs, it bails before any
        dispatch/DB work — these daemon threads are untracked and not
        awaited by _llm_executor.drain(), so the closing flag is their only
        guard against reading a database die() is tearing down.
        """
        msg = mocker.MagicMock()
        captured: dict = {}

        class _FakeThread:
            def __init__(self, target, name=None, daemon=None):  # type: ignore[no-untyped-def]
                captured["target"] = target

            def start(self) -> None:
                pass

        mocker.patch("llm.plugin.world.SupyThread", _FakeThread)
        plugin._dispatch_with_verse_routing = mocker.MagicMock()
        plugin._llm_executor = mocker.MagicMock(closing=True)

        preflight = PreflightResult(False, Caller(Identity("u", None), "#c"))
        plugin._dispatch_addressed_async(mock_irc, msg, "hi", preflight, entry_route="addressed")

        captured["target"]()  # run the worker
        plugin._dispatch_with_verse_routing.assert_not_called()

    def test_pm_duplicate_filter_is_per_sender(
        self, plugin, mocker: MockerFixture, mock_irc: MagicMock
    ) -> None:
        """GIVEN alice and bob both PM "hi" within the dedup window THEN both
        get a dispatch. In a PM msg.args[0] is the bot's own nick, so keying
        the filter on it put every PM sender in one bucket; the reply target
        is the sender."""
        from llm.plugin import PreflightResult

        started: list[str] = []

        class _FakeThread:
            def __init__(self, target, name=None, daemon=None):  # type: ignore[no-untyped-def]
                pass

            def start(self) -> None:
                started.append("x")

        mocker.patch("llm.plugin.world.SupyThread", _FakeThread)
        for nick in ("alice", "bob", "alice"):
            msg = mocker.MagicMock(nick=nick)
            preflight = PreflightResult(blocked=False, caller=Caller(Identity(nick, None), None))
            plugin._dispatch_addressed_async(mock_irc, msg, "hi", preflight, entry_route="pm")

        # alice, bob; alice's repeat is the duplicate.
        assert len(started) == 2


# =========================================================================
# Integration tests (real DB + context)
# =========================================================================


class TestMetaIntegration:
    """End-to-end integration tests for the meta feature."""

    def test_set_instruction_via_meta(self, mocker: MockerFixture) -> None:
        """GIVEN user says 'always respond in haiku' WHEN routed through meta
        THEN instruction is saved and confirmation returned."""
        from llm.context import ContextConfig, ConversationContext
        from llm.persistence import LLMDatabase

        db = LLMDatabase(":memory:")
        config = ContextConfig(
            max_messages=20,
            timeout_minutes=5,
            channel_max_messages=10,
        )
        context = ConversationContext(config)

        svc, _plugin = self._make_service(mocker)

        # Simulate: tool call -> set_instruction -> text response
        tool_call = make_tool_call(
            "set_instruction", {"text": "always respond in haiku"}, call_id="call_1"
        )
        first_response = make_completion_response(None, tool_calls=[tool_call])
        second_response = make_completion_response("Done \u2014 I'll respond in haiku.")

        mocker.patch(
            "llm.service.provider.chat",
            side_effect=[first_response, second_response],
        )

        result = svc.assistant_completion(
            route_profile=PROFILE_REMIND_ACTION,
            prompt="always respond in haiku",
            nick="testuser",
            account="testuser",
            channel="#test",
            db=db,
            context=context,
            bot_nick="VibeBot",
        )
        assert "haiku" in result.content.lower()
        assert db.get_instruction("testuser") == "always respond in haiku"

        db.close()

    def test_list_and_delete_memories_via_meta(self, mocker: MockerFixture) -> None:
        """GIVEN user has memories WHEN meta deletes by topic THEN removed."""
        from llm.context import ContextConfig, ConversationContext
        from llm.persistence import LLMDatabase

        db = LLMDatabase(":memory:")
        config = ContextConfig(
            max_messages=20,
            timeout_minutes=5,
            channel_max_messages=10,
        )
        context = ConversationContext(config)

        id1 = db.save_memory("testuser", "likes cats", "#test")
        id2 = db.save_memory("testuser", "owns two cats", "#test")
        _id3 = db.save_memory("testuser", "uses vim", "#test")

        svc, _plugin = self._make_service(mocker)

        # Simulate: list -> delete two cat memories -> confirmation
        list_call = make_tool_call("list_memories", call_id="call_list")
        r1 = make_completion_response(None, tool_calls=[list_call])

        del_call_1 = make_tool_call("delete_memory", {"id": id1}, call_id="call_del_1")
        del_call_2 = make_tool_call("delete_memory", {"id": id2}, call_id="call_del_2")
        r2 = make_completion_response(None, tool_calls=[del_call_1, del_call_2])

        r3 = make_completion_response("Deleted 2 memories about cats.")

        mocker.patch(
            "llm.service.provider.chat",
            side_effect=[r1, r2, r3],
        )

        result = svc.assistant_completion(
            route_profile=PROFILE_REMIND_ACTION,
            prompt="delete any memories about cats",
            nick="testuser",
            account="testuser",
            channel="#test",
            db=db,
            context=context,
            bot_nick="VibeBot",
        )
        assert "cat" in result.content.lower()

        remaining = db.get_memories("testuser")
        assert len(remaining) == 1
        assert remaining[0].fact == "uses vim"

        db.close()

    def test_usage_logged_with_meta_command(self) -> None:
        """GIVEN a meta call WHEN usage logged THEN recorded as 'meta'."""
        from llm.persistence import LLMDatabase

        db = LLMDatabase(":memory:")
        db.log_usage(
            nick="testuser",
            channel="#test",
            command="meta",
            model="gemini/gemini-2.0-flash",
            prompt_tokens=100,
            completion_tokens=50,
            cost=0.001,
        )

        summary = db.get_usage_summary_for_nick("testuser")
        assert summary.total_requests == 1

        db.close()

    def test_get_usage_via_meta(self, mocker: MockerFixture) -> None:
        """GIVEN user asks about usage WHEN meta handles it THEN returns stats."""
        from llm.persistence import LLMDatabase

        db = LLMDatabase(":memory:")
        db.log_usage("testuser", "#test", "ask", "gpt-4", 100, 50, 0.01)
        db.log_usage("testuser", "#test", "ask", "gpt-4", 200, 100, 0.02)

        svc, _plugin = self._make_service(mocker)

        tool_call = make_tool_call("get_usage", call_id="call_usage")
        first_response = make_completion_response(None, tool_calls=[tool_call])
        second_response = make_completion_response("You've made 2 requests costing $0.03.")

        mocker.patch(
            "llm.service.provider.chat",
            side_effect=[first_response, second_response],
        )

        from llm.context import ContextConfig, ConversationContext

        result = svc.assistant_completion(
            prompt="how much have I used?",
            nick="testuser",
            channel="#test",
            db=db,
            context=ConversationContext(
                ContextConfig(
                    max_messages=20,
                    timeout_minutes=5,
                    channel_max_messages=10,
                )
            ),
            bot_nick="VibeBot",
        )
        assert "2" in result.content
        db.close()

    def test_cleanup_via_meta(self, mocker: MockerFixture) -> None:
        """GIVEN cleanup_fn callable WHEN meta calls it THEN cleanup runs."""
        svc, _plugin = self._make_service(mocker)

        tool_call = make_tool_call("cleanup_memories", call_id="call_cleanup")
        first_response = make_completion_response(None, tool_calls=[tool_call])
        second_response = make_completion_response("Cleaned up your memories.")

        mocker.patch(
            "llm.service.provider.chat",
            side_effect=[first_response, second_response],
        )

        from llm.context import ContextConfig, ConversationContext

        cleanup_fn = mocker.MagicMock(return_value="Before: 5 | dropped: 1 | after: 4")

        svc.assistant_completion(
            route_profile=PROFILE_REMIND_ACTION,
            prompt="clean up my memories",
            nick="testuser",
            account="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=ConversationContext(
                ContextConfig(
                    max_messages=20,
                    timeout_minutes=5,
                    channel_max_messages=10,
                )
            ),
            bot_nick="VibeBot",
            cleanup_fn=cleanup_fn,
        )
        cleanup_fn.assert_called_once()

    def test_set_reminder_via_meta(self, mocker: MockerFixture) -> None:
        """GIVEN set_reminder callable WHEN meta calls it THEN reminder set."""
        svc, _plugin = self._make_service(mocker)

        tool_call = make_tool_call(
            "set_reminder", {"text": "deploy in 2 hours"}, call_id="call_remind"
        )
        first_response = make_completion_response(None, tool_calls=[tool_call])
        second_response = make_completion_response("Reminder set: deploy (in 2 hours).")

        mocker.patch(
            "llm.service.provider.chat",
            side_effect=[first_response, second_response],
        )

        from llm.context import ContextConfig, ConversationContext

        set_reminder_fn = mocker.MagicMock(return_value="I'll remind you in 2 hours")

        svc.assistant_completion(
            prompt="remind me to deploy in 2 hours",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=ConversationContext(
                ContextConfig(
                    max_messages=20,
                    timeout_minutes=5,
                    channel_max_messages=10,
                )
            ),
            bot_nick="VibeBot",
            set_reminder_fn=set_reminder_fn,
        )
        set_reminder_fn.assert_called_once_with("deploy in 2 hours")

    @staticmethod
    def _make_service(mocker: MockerFixture) -> tuple:
        """Create an LLMService with meta config defaults."""
        plugin = mocker.MagicMock()
        plugin.registryValue = mocker.Mock(
            side_effect=make_registry_side_effect({"assistantModel": "gpt-4"})
        )
        plugin.log = mocker.Mock()
        return LLMService(plugin), plugin


class TestToolResult:
    """Tests for the ToolResult frozen dataclass."""

    def test_defaults(self) -> None:
        """GIVEN ToolResult with only content WHEN created THEN defaults are correct."""
        result = ToolResult(content="search result")
        assert result.content == "search result"
        assert result.grounding_used is False
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0
        assert result.cost == 0.0

    def test_all_fields_set(self) -> None:
        """GIVEN ToolResult with all fields WHEN created THEN all values stored."""
        result = ToolResult(
            content="fetched page",
            grounding_used=True,
            prompt_tokens=200,
            completion_tokens=100,
            cost=0.005,
        )
        assert result.content == "fetched page"
        assert result.grounding_used is True
        assert result.prompt_tokens == 200
        assert result.completion_tokens == 100
        assert result.cost == 0.005

    def test_frozen(self) -> None:
        """GIVEN a ToolResult WHEN attempting mutation THEN raises FrozenInstanceError."""
        result = ToolResult(content="immutable")
        with pytest.raises(AttributeError):
            result.content = "changed"  # type: ignore[misc]


class TestChatToolSurfaceStaysSmall:
    """The advertised chat surface is a correctness budget, not housekeeping.

    grok-4-1-fast-reasoning starts returning empty completions past ~25 tools,
    and a non-reasoning model picking one tool out of twenty sometimes picks
    none and invents an answer instead -- the 2026-08-01 fabricated image URL.
    Verse was trimmed for this reason long ago; chat kept all twenty until the
    bookkeeping tools were hidden.
    """

    def test_no_bookkeeping_tool_is_offered_in_chat(self) -> None:
        """Each of these duplicates a command the user can type directly."""
        names = {t["function"]["name"] for t in get_tools_for_profile("chat")}
        offered = sorted(names & _BOOKKEEPING_TOOLS)
        assert not offered, f"bookkeeping tools leaked back into chat: {offered}"

    def test_chat_keeps_the_tools_that_do_real_work(self) -> None:
        """Generation and research must survive any future trimming."""
        names = {t["function"]["name"] for t in get_tools_for_profile("chat")}
        for tool in ("search_web", "fetch_url", "generate_image", "generate_code"):
            assert tool in names, f"{tool} missing from chat profile"

    def test_save_memory_survives_but_the_readers_do_not(self) -> None:
        """Memories are learned automatically; they are not administered by chat.

        save_memory is the one write the background extractor cannot stand in
        for, because an explicit "remember this" should stick immediately
        rather than wait for candidate reinforcement.
        """
        names = {t["function"]["name"] for t in get_tools_for_profile("chat")}
        assert "save_memory" in names
        assert not names & {"list_memories", "update_memory", "delete_memory"}

    def test_chat_surface_stays_under_budget(self) -> None:
        """A ceiling, so the surface cannot creep back one tool at a time.

        12, not 10, since list/cancel/cancel_all_pending_tasks came back on
        2026-09-25: without them chat faked every cancel it was asked for.
        """
        count = len(get_tools_for_profile("chat"))
        assert count <= 12, f"chat advertises {count} tools; keep the surface small"


class TestToolSpecVisibility:
    """GIVEN tool specs WHEN inspected THEN visibility and capability are correct."""

    def test_search_web_visible_in_chat(self) -> None:
        specs = {s.name: s for s in ASSISTANT_TOOL_SPECS}
        assert "chat" in specs["search_web"].visible_in

    def test_search_web_visible_in_code(self) -> None:
        specs = {s.name: s for s in ASSISTANT_TOOL_SPECS}
        assert "code" in specs["search_web"].visible_in

    def test_search_web_not_visible_in_draw(self) -> None:
        specs = {s.name: s for s in ASSISTANT_TOOL_SPECS}
        assert "draw" not in specs["search_web"].visible_in

    def test_fetch_url_visible_in_chat_code_and_remind_action(self) -> None:
        specs = {s.name: s for s in ASSISTANT_TOOL_SPECS}
        assert specs["fetch_url"].visible_in == frozenset(
            {"chat", "verse", "code", "remind_action"}
        )

    def test_generate_code_capability_is_llm_code(self) -> None:
        specs = {s.name: s for s in ASSISTANT_TOOL_SPECS}
        assert specs["generate_code"].capability == "llm.code"

    def test_generate_code_visible_in_chat_and_code_and_remind_action(self) -> None:
        specs = {s.name: s for s in ASSISTANT_TOOL_SPECS}
        assert specs["generate_code"].visible_in == frozenset(
            {"chat", "verse", "code", "remind_action"}
        )

    def test_generate_image_visible_in_chat_draw_and_remind_action(self) -> None:
        specs = {s.name: s for s in ASSISTANT_TOOL_SPECS}
        assert specs["generate_image"].visible_in == frozenset(
            {"chat", "verse", "draw", "remind_action"}
        )

    def test_verse_profile_is_strict_subset_of_chat(self) -> None:
        """Verse drops scheduling/usage/instruction tools that drown the model.

        Empirically xai/grok-4-1-fast-reasoning starts emitting empty
        completions once the advertised tool count climbs past ~25 (4
        empty-response incidents on 2026-05-10 alone, more than any
        prior day in 30d). Verse mode is in-character roleplay — the
        scheduling/reminder/usage/instruction tools have no in-world use
        but still bloat every prompt with their schemas. Trim them.
        """
        chat_tools = {t["function"]["name"] for t in get_tools_for_profile("chat")}
        verse_tools = {t["function"]["name"] for t in get_tools_for_profile("verse")}
        assert verse_tools < chat_tools

    def test_verse_profile_excludes_scheduling_and_meta_tools(self) -> None:
        """Tools that have no in-character use are hidden from verse mode."""
        names = {t["function"]["name"] for t in get_tools_for_profile("verse")}
        for tool in (
            "set_reminder",
            "list_pending_tasks",
            "cancel_pending_task",
            "cancel_all_pending_tasks",
            "schedule_llm_task",
            "get_usage",
            "get_channel_usage",
            "forget_context",
            "cleanup_memories",
            "clear_memories",
            "clear_instruction",
            "set_instruction",
        ):
            assert tool not in names, f"{tool} should not be visible in verse profile"

    def test_verse_profile_keeps_research_and_creative_tools(self) -> None:
        """Verse keeps what the bot uses in-character: research, creative
        output, and the one memory WRITE.

        Memory management (list/delete/update) is not in-character and is
        reachable via @memories, so it is hidden along with the rest of
        ``_BOOKKEEPING_TOOLS``. save_memory stays because the extractor
        cannot substitute for an explicit "remember this".
        """
        names = {t["function"]["name"] for t in get_tools_for_profile("verse")}
        for tool in (
            "search_web",
            "fetch_url",
            "generate_image",
            "generate_code",
            "save_memory",
        ):
            assert tool in names, f"{tool} missing from verse profile"
        for tool in ("list_memories", "delete_memory", "update_memory"):
            assert tool not in names, f"{tool} should be hidden from verse"

    def test_profile_tools_remind_action_includes_search_fetch_code_image(self) -> None:
        """Action reminders need the union of @ask + @draw tool surfaces."""
        tools = get_tools_for_profile("remind_action")
        names = {t["function"]["name"] for t in tools}
        for required in ("search_web", "fetch_url", "generate_code", "generate_image"):
            assert required in names, f"{required} missing from remind_action profile"
        # Sanity: also includes ordinary chat tools (defaults to chat+remind_action)
        assert "list_pending_tasks" in names
        assert "set_reminder" in names

    def test_profile_tools_chat_includes_search(self) -> None:
        tools = get_tools_for_profile("chat")
        names = {t["function"]["name"] for t in tools}
        assert "search_web" in names
        assert "generate_code" in names
        assert "generate_image" in names

    def test_profile_tools_draw_includes_generate_image(self) -> None:
        tools = get_tools_for_profile("draw")
        names = {t["function"]["name"] for t in tools}
        assert "generate_image" in names

    def test_profile_tools_draw_excludes_search(self) -> None:
        tools = get_tools_for_profile("draw")
        names = {t["function"]["name"] for t in tools}
        assert "search_web" not in names
        assert "generate_code" not in names

    def test_profile_tools_code_includes_search_and_code(self) -> None:
        tools = get_tools_for_profile("code")
        names = {t["function"]["name"] for t in tools}
        assert "search_web" in names
        assert "fetch_url" in names
        assert "generate_code" in names

    def test_set_reminder_visible_in_chat_and_remind_action(self) -> None:
        """set_reminder routes through chat (deferred entry point) and remind_action
        (self-rescheduling). Immediate-execution profiles (@draw, @code) defer via
        chat using @remind, so they should NOT expose the reminder tools."""
        for profile in ("chat", "remind_action"):
            tools = get_tools_for_profile(profile)
            names = {t["function"]["name"] for t in tools}
            assert "set_reminder" in names, f"set_reminder missing from {profile} profile"

    def test_reminder_tools_not_visible_in_draw_profile(self) -> None:
        """@draw is immediate-execution; deferred draws go via @remind (chat profile)."""
        tools = get_tools_for_profile("draw")
        names = {t["function"]["name"] for t in tools}
        for tool in (
            "set_reminder",
            "list_pending_tasks",
            "cancel_pending_task",
            "cancel_all_pending_tasks",
        ):
            assert tool not in names, f"{tool} should not be visible in draw profile"

    def test_reminder_tools_not_visible_in_code_profile(self) -> None:
        """@code is immediate-execution; deferred code generation goes via @remind."""
        tools = get_tools_for_profile("code")
        names = {t["function"]["name"] for t in tools}
        for tool in (
            "set_reminder",
            "list_pending_tasks",
            "cancel_pending_task",
            "cancel_all_pending_tasks",
        ):
            assert tool not in names, f"{tool} should not be visible in code profile"


class TestScheduleLlmTaskFamily:
    """Phase 2 Task 3 / C1 — schedule_llm_task tool registration."""

    def test_assistant_tools_includes_schedule_llm_task_family(self) -> None:
        names = {t["function"]["name"] for t in ASSISTANT_TOOLS}
        assert "schedule_llm_task" in names
        # Listing/cancelling for both reminders and scheduled tasks now
        # goes through the unified pending-task tool surface.
        assert "list_pending_tasks" in names
        assert "cancel_pending_task" in names
        assert "cancel_all_pending_tasks" in names

        by_name = {t["function"]["name"]: t for t in ASSISTANT_TOOLS}
        sch = by_name["schedule_llm_task"]
        # Description must call out the @ask-with-tools shape and contrast
        # with set_reminder so the LLM picks the right tool.
        desc = sch["function"]["description"].lower()
        assert "@ask" in desc or "ask " in desc
        assert "set_reminder" in desc
        assert "tool" in desc

        params = sch["function"]["parameters"]
        assert "when_natural" in params["properties"]
        assert "prompt" in params["properties"]
        assert set(params["required"]) >= {"when_natural", "prompt"}

    def test_schedule_llm_task_specs_overrides_applied(self) -> None:
        """C2: ToolSpec overrides give schedule_llm_task require_account=True;
        list/cancel inherit defaults (llm.ask, no account). All three are
        hidden from verse and visible in chat, which fakes a cancel it
        cannot perform — see _PROFILE_EXCLUDED_TOOLS."""
        from llm.assistant import ASSISTANT_TOOL_REGISTRY

        sch = ASSISTANT_TOOL_REGISTRY["schedule_llm_task"]
        assert sch.capability == "llm.ask"
        assert sch.require_account is True
        assert sch.visible_in == frozenset({"chat", "remind_action"})

        lst = ASSISTANT_TOOL_REGISTRY["list_pending_tasks"]
        assert lst.capability == "llm.ask"
        assert lst.require_account is False
        assert lst.visible_in == frozenset({"chat", "remind_action"})

        can = ASSISTANT_TOOL_REGISTRY["cancel_pending_task"]
        assert can.capability == "llm.ask"
        assert can.require_account is False
        assert can.visible_in == frozenset({"chat", "remind_action"})

    def test_executor_accepts_pending_task_fns(self, mocker: MockerFixture) -> None:
        """C3: AssistantToolExecutor accepts the unified pending-task fn kwargs."""
        schedule_fn = mocker.MagicMock()
        list_fn = mocker.MagicMock()
        cancel_fn = mocker.MagicMock()
        cancel_all_fn = mocker.MagicMock()

        ex = make_executor(
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            nick="n",
            channel="#t",
            capabilities=frozenset({"llm.ask"}),
            account="acct",
            schedule_llm_task_fn=schedule_fn,
            list_pending_tasks_fn=list_fn,
            cancel_pending_task_fn=cancel_fn,
            cancel_all_pending_tasks_fn=cancel_all_fn,
        )
        assert ex._schedule_llm_task_fn is schedule_fn
        assert ex._list_pending_tasks_fn is list_fn
        assert ex._cancel_pending_task_fn is cancel_fn
        assert ex._cancel_all_pending_tasks_fn is cancel_all_fn

    def test_tool_schedule_llm_task_calls_callback_and_returns_json(
        self, mocker: MockerFixture
    ) -> None:
        import json

        schedule_fn = mocker.MagicMock(
            return_value={
                "status": "ok",
                "event_name": "llm_task_abc",
                "fire_at": 1700000000.0,
                "message": "Scheduled.",
            }
        )

        ex = make_executor(
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            nick="n",
            channel="#t",
            capabilities=frozenset({"llm.ask"}),
            account="acct",
            schedule_llm_task_fn=schedule_fn,
        )

        out = ex.execute(
            "schedule_llm_task",
            {"when_natural": "in 60s", "prompt": "ping me"},
        )
        parsed = json.loads(out.content)
        assert parsed["status"] == "ok"
        assert parsed["event_name"] == "llm_task_abc"
        schedule_fn.assert_called_once_with(
            when_natural="in 60s", prompt="ping me", reply_target=None
        )

    def test_tool_list_pending_tasks_returns_merged(self, mocker: MockerFixture) -> None:
        import json

        list_fn = mocker.MagicMock(
            return_value=[
                {
                    "kind": "reminder",
                    "id": "abc123",
                    "channel": "#t",
                    "description": "check build",
                },
                {
                    "kind": "scheduled_task",
                    "id": "llm_task_ev2",
                    "when": "2026-05-09T13:00:00Z",
                    "channel": "#t",
                    "description": "weekly digest",
                    "recurrence": "FREQ=WEEKLY;BYDAY=MO;BYHOUR=9",
                },
            ]
        )
        ex = make_executor(
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            nick="n",
            channel="#t",
            capabilities=frozenset({"llm.ask"}),
            account="acct",
            list_pending_tasks_fn=list_fn,
        )
        out = ex.execute("list_pending_tasks", {})
        parsed = json.loads(out.content)
        assert len(parsed["tasks"]) == 2
        kinds = {t["kind"] for t in parsed["tasks"]}
        assert kinds == {"reminder", "scheduled_task"}

    def test_tool_cancel_pending_task_passes_id(self, mocker: MockerFixture) -> None:
        cancel_fn = mocker.MagicMock(
            return_value={
                "status": "ok",
                "kind": "scheduled_task",
                "id": "llm_task_abc",
                "message": "Cancelled.",
            }
        )
        ex = make_executor(
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            nick="n",
            channel="#t",
            capabilities=frozenset({"llm.ask"}),
            account="acct",
            cancel_pending_task_fn=cancel_fn,
        )
        out = ex.execute("cancel_pending_task", {"id": "llm_task_abc"})
        assert "ok" in out.content.lower()
        cancel_fn.assert_called_once_with("llm_task_abc")


class TestPersonalDataToolsNeedAccount:
    """Memory and instruction tools key on account-or-nick, so for an
    unauthenticated caller the nick could be someone else's account. They
    refuse with a tool error, matching @memories and @instruct."""

    @pytest.mark.parametrize(
        ("tool", "args", "db_call"),
        [
            ("save_memory", {"text": "likes tea"}, "save_memory"),
            ("list_memories", {}, "get_memories"),
            ("delete_memory", {"id": 1}, "delete_memory"),
            ("update_memory", {"id": 1, "text": "x"}, "update_memory"),
            ("clear_memories", {}, "delete_all_memories"),
            ("set_instruction", {"text": "be brief"}, "save_instruction"),
            ("clear_instruction", {}, "delete_instruction"),
        ],
    )
    def test_accountless_reminder_action_is_refused(
        self, mocker: MockerFixture, tool, args, db_call
    ):
        """GIVEN a reminder-action fire for an accountless owner (the route that
        offers every one of these tools) WHEN a personal tool runs THEN it
        refuses and touches no data."""
        db = mocker.MagicMock()
        executor = make_executor(
            db=db,
            context=mocker.MagicMock(),
            nick="rdrake",
            channel="#t",
            account=None,
            route_profile=PROFILE_REMIND_ACTION,
        )
        result = executor.execute(tool, args)
        assert "authenticated account" in result.content
        getattr(db, db_call).assert_not_called()

    def test_accountless_chat_save_memory_is_refused(self, mocker: MockerFixture):
        """save_memory is the one personal tool chat still offers."""
        db = mocker.MagicMock()
        executor = make_executor(
            db=db,
            context=mocker.MagicMock(),
            nick="rdrake",
            channel="#t",
            account=None,
            route_profile=PROFILE_CHAT,
        )
        assert "authenticated account" in executor.execute("save_memory", {"text": "x"}).content
        db.save_memory.assert_not_called()

    def test_cleanup_memories_refused_without_account(self, mocker: MockerFixture):
        cleanup = mocker.MagicMock()
        executor = make_executor(
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            nick="rdrake",
            channel="#t",
            account=None,
            cleanup_fn=cleanup,
        )
        assert "authenticated account" in executor.execute("cleanup_memories", {}).content
        cleanup.assert_not_called()

    def test_authenticated_caller_saves(self, mocker: MockerFixture):
        db = mocker.MagicMock()
        db.save_memory.return_value = 7
        executor = make_executor(
            db=db,
            context=mocker.MagicMock(),
            nick="rdrake",
            channel="#t",
            account="rdrake",
            route_profile=PROFILE_CHAT,
        )
        assert '"ok"' in executor.execute("save_memory", {"text": "likes tea"}).content
        db.save_memory.assert_called_once_with("rdrake", "likes tea", "#t")


class TestExecutorCoverageGaps:
    """Targeted tests for branches not exercised by the main suite."""

    def test_denial_reason_require_account_without_account(self) -> None:
        """ToolSpec.denial_reason rejects when require_account=True and account is None."""
        from llm.assistant import ASSISTANT_TOOL_REGISTRY

        spec = ASSISTANT_TOOL_REGISTRY["schedule_llm_task"]
        reason = spec.denial_reason(
            route_profile="chat",
            capabilities=frozenset({"llm.ask"}),
            account=None,
        )
        assert reason is not None
        assert "authenticated account" in reason

    def test_execute_missing_handler_returns_error(
        self, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """execute() falls through to the missing-handler branch when handler_name is bogus."""
        from llm.assistant import ASSISTANT_TOOL_REGISTRY, ToolSpec

        bogus = ToolSpec(
            name="bogus_handler_tool",
            schema={"name": "bogus_handler_tool", "description": "x", "parameters": {}},
            handler_name="_tool_does_not_exist",
            capability="llm.ask",
        )
        monkeypatch.setitem(ASSISTANT_TOOL_REGISTRY, "bogus_handler_tool", bogus)

        ex = make_executor(
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            nick="n",
            channel="#t",
        )
        result = ex.execute("bogus_handler_tool", {})
        assert "Unknown tool implementation" in result.content

    @pytest.mark.parametrize(
        "tool_name,args",
        [
            ("set_instruction", {"nick": "someone", "text": "x"}),
            ("clear_instruction", {"nick": "someone"}),
            ("save_memory", {"nick": "someone", "text": "x"}),
            ("delete_memory", {"nick": "someone", "id": 1}),
            ("update_memory", {"nick": "someone", "id": 1, "text": "x"}),
        ],
    )
    def test_per_tool_owner_only_denials(
        self,
        mocker: MockerFixture,
        tool_name: str,
        args: dict,
    ) -> None:
        """Non-owner targeting another nick is denied for owner-gated tools."""
        ex = make_executor(
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            nick="regular",
            channel="#t",
        )
        result = ex.execute(tool_name, args)
        assert "owner" in result.content.lower()

    def test_forget_context_when_nothing_to_clear(self, mocker: MockerFixture) -> None:
        """forget_context returns the no-op message when context.clear() returns False."""
        ctx = mocker.MagicMock()
        ctx.clear.return_value = False
        ex = make_executor(
            db=mocker.MagicMock(),
            context=ctx,
            nick="n",
            channel="#t",
        )
        result = ex.execute("forget_context", {})
        assert "No context to clear" in result.content

    def test_cleanup_memories_error_result_is_routed_as_error(self, mocker: MockerFixture) -> None:
        """cleanup_fn returning ok=False produces an error envelope."""
        cleanup_fn = mocker.MagicMock(
            return_value=ToolCallbackResult(False, "Cleanup failed: db error")
        )
        ex = make_executor(
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            nick="n",
            channel="#t",
            cleanup_fn=cleanup_fn,
        )
        result = ex.execute("cleanup_memories", {})
        assert '"error"' in result.content
        assert "Cleanup failed" in result.content

    def test_cleanup_memories_success_with_error_word_in_message_is_ok(
        self, mocker: MockerFixture
    ) -> None:
        """A success message containing the word 'errors' is NOT misclassified.

        Pinning the new behavior: classification comes from ``ok``, not
        from substring-sniffing the message.
        """
        cleanup_fn = mocker.MagicMock(
            return_value=ToolCallbackResult(True, "Removed 3 errors from your memories")
        )
        ex = make_executor(
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            nick="n",
            channel="#t",
            cleanup_fn=cleanup_fn,
        )
        result = ex.execute("cleanup_memories", {})
        assert '"ok"' in result.content

    @pytest.mark.parametrize(
        "tool_name,args",
        [
            ("list_pending_tasks", {}),
            ("set_reminder", {"text": "in 1h ping"}),
            ("cancel_pending_task", {"id": "abc"}),
            ("cancel_all_pending_tasks", {}),
        ],
    )
    def test_pending_task_tools_unavailable_when_no_fn(
        self, mocker: MockerFixture, tool_name: str, args: dict
    ) -> None:
        """Pending-task tools return 'not available' when their callback is None."""
        ex = make_executor(
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            nick="n",
            channel="#t",
        )
        result = ex.execute(tool_name, args)
        assert "not available" in result.content.lower()

    def test_set_reminder_error_result_is_routed_as_error(self, mocker: MockerFixture) -> None:
        """set_reminder_fn returning ok=False becomes an error envelope."""
        set_fn = mocker.MagicMock(return_value=ToolCallbackResult(False, "Could not parse time."))
        ex = make_executor(
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            nick="n",
            channel="#t",
            set_reminder_fn=set_fn,
        )
        result = ex.execute("set_reminder", {"text": "garbage"})
        assert '"error"' in result.content
        assert "Could not parse" in result.content

    def test_set_reminder_success_with_failure_word_is_ok(self, mocker: MockerFixture) -> None:
        """A success message containing 'failed' is NOT misclassified.

        Pinning the new behavior: classification comes from ``ok``, not
        from substring-sniffing the message.
        """
        set_fn = mocker.MagicMock(
            return_value=ToolCallbackResult(True, "Reminder set; previous failed reminders cleared")
        )
        ex = make_executor(
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            nick="n",
            channel="#t",
            set_reminder_fn=set_fn,
        )
        result = ex.execute("set_reminder", {"text": "in 1h ping"})
        assert '"ok"' in result.content

    def test_schedule_llm_task_unconfigured(self, mocker: MockerFixture) -> None:
        """schedule_llm_task without a callback returns 'not configured'."""
        ex = make_executor(
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            nick="n",
            channel="#t",
            capabilities=frozenset({"llm.ask"}),
            account="acct",
        )
        result = ex.execute(
            "schedule_llm_task",
            {"when_natural": "in 1h", "prompt": "ping"},
        )
        assert "not configured" in result.content.lower()

    @pytest.mark.parametrize(
        "args,expected_substring",
        [
            ({"when_natural": "", "prompt": "ping"}, "when_natural is required"),
            ({"when_natural": "in 1h", "prompt": ""}, "prompt is required"),
        ],
    )
    def test_schedule_llm_task_validates_required_args(
        self,
        mocker: MockerFixture,
        args: dict,
        expected_substring: str,
    ) -> None:
        """schedule_llm_task rejects empty when_natural / prompt before calling fn."""
        schedule_fn = mocker.MagicMock()
        ex = make_executor(
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            nick="n",
            channel="#t",
            capabilities=frozenset({"llm.ask"}),
            account="acct",
            schedule_llm_task_fn=schedule_fn,
        )
        result = ex.execute("schedule_llm_task", args)
        assert expected_substring in result.content
        schedule_fn.assert_not_called()

    def test_cancel_pending_task_requires_id(self, mocker: MockerFixture) -> None:
        """cancel_pending_task rejects empty id before calling fn."""
        cancel_fn = mocker.MagicMock()
        ex = make_executor(
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            nick="n",
            channel="#t",
            capabilities=frozenset({"llm.ask"}),
            cancel_pending_task_fn=cancel_fn,
        )
        result = ex.execute("cancel_pending_task", {"id": ""})
        assert "id is required" in result.content
        cancel_fn.assert_not_called()

    def test_generate_image_schema_offers_three_shapes(self) -> None:
        """The planner can only choose what the schema lists; xAI's map is keyed on it."""
        from llm.assistant import ASSISTANT_TOOLS

        spec = next(t for t in ASSISTANT_TOOLS if t["function"]["name"] == "generate_image")
        aspect = spec["function"]["parameters"]["properties"]["aspect"]
        assert aspect["enum"] == ["portrait", "landscape", "square"]
        assert "aspect" not in spec["function"]["parameters"]["required"]

    def test_generate_image_unavailable_when_no_draw_fn(self, mocker: MockerFixture) -> None:
        """generate_image without a draw_fn returns 'not available'."""
        ex = make_executor(
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            nick="n",
            channel="#t",
            capabilities=frozenset({"llm.ask", "llm.draw"}),
            account="acct",
        )
        result = ex.execute("generate_image", {"prompt": "a cat"})
        assert "not available" in result.content.lower()

    def test_generate_image_rejects_empty_prompt(self, mocker: MockerFixture) -> None:
        """generate_image rejects whitespace-only prompts before calling draw_fn."""
        draw_fn = mocker.MagicMock()
        ex = make_executor(
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            nick="n",
            channel="#t",
            capabilities=frozenset({"llm.ask", "llm.draw"}),
            account="acct",
            draw_fn=draw_fn,
        )
        result = ex.execute("generate_image", {"prompt": "   "})
        assert "prompt is required" in result.content.lower()
        draw_fn.assert_not_called()

    def test_generate_image_propagates_draw_fn_error(self, mocker: MockerFixture) -> None:
        """draw_fn returning ok=False is mapped to an error envelope."""
        draw_fn = mocker.MagicMock(return_value=ToolCallbackResult(False, "Error: quota exceeded"))
        ex = make_executor(
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            nick="n",
            channel="#t",
            capabilities=frozenset({"llm.ask", "llm.draw"}),
            account="acct",
            draw_fn=draw_fn,
        )
        result = ex.execute("generate_image", {"prompt": "a cat"})
        assert '"error"' in result.content
        assert "quota exceeded" in result.content

    def test_generate_image_success(self, mocker: MockerFixture) -> None:
        """draw_fn returning ok=True produces an ok envelope."""
        draw_fn = mocker.MagicMock(
            return_value=ToolCallbackResult(True, "https://example.com/cat.png")
        )
        ex = make_executor(
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            nick="n",
            channel="#t",
            capabilities=frozenset({"llm.ask", "llm.draw"}),
            account="acct",
            draw_fn=draw_fn,
        )
        result = ex.execute("generate_image", {"prompt": "a cat"})
        assert '"ok"' in result.content
        assert "example.com/cat.png" in result.content

    def test_generate_image_passes_the_aspect_to_draw_fn(self, mocker: MockerFixture) -> None:
        """GIVEN the model chose a shape WHEN the tool runs THEN draw_fn gets it by name."""
        draw_fn = mocker.MagicMock(return_value=ToolCallbackResult(True, "https://x/y.png"))
        ex = make_executor(
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            nick="n",
            channel="#t",
            capabilities=frozenset({"llm.ask", "llm.draw"}),
            account="acct",
            draw_fn=draw_fn,
        )
        ex.execute("generate_image", {"prompt": "a poster", "aspect": "portrait"})
        draw_fn.assert_called_once_with("a poster", aspect="portrait")

    def test_generate_image_without_an_aspect_sends_none(self, mocker: MockerFixture) -> None:
        """GIVEN no aspect (or garbage) WHEN the tool runs THEN draw_fn gets None."""
        draw_fn = mocker.MagicMock(return_value=ToolCallbackResult(True, "https://x/y.png"))
        ex = make_executor(
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            nick="n",
            channel="#t",
            capabilities=frozenset({"llm.ask", "llm.draw"}),
            account="acct",
            draw_fn=draw_fn,
        )
        ex.execute("generate_image", {"prompt": "a cat", "aspect": 7})
        draw_fn.assert_called_once_with("a cat", aspect=None)

    def test_generate_image_does_not_misclassify_ok_message_with_error_word(
        self, mocker: MockerFixture
    ) -> None:
        """A success message containing 'Error' is NOT mapped to error.

        Pinning the new behavior: classification comes from ``ok``, not
        from sniffing the message string for ``Error``.
        """
        draw_fn = mocker.MagicMock(
            return_value=ToolCallbackResult(True, "Error-free image at https://x/y.png")
        )
        ex = make_executor(
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            nick="n",
            channel="#t",
            capabilities=frozenset({"llm.ask", "llm.draw"}),
            account="acct",
            draw_fn=draw_fn,
        )
        result = ex.execute("generate_image", {"prompt": "a cat"})
        assert '"ok"' in result.content


class TestAssistantCompletionReadsModelKeyFromProfiles:
    """assistant_completion must read the model setting via PROFILES[route].

    Swapping the PROFILES entry with a sentinel-keyed Profile and asserting
    the sentinel key flows to plugin.registryValue() pins that the
    migration is wired up — a future regression that hardcodes
    'assistantModel' would break these tests. The API key is not part of
    Profile — it is resolved from the model at the completion boundary, not
    looked up per-route — so this class covers model_setting only.
    """

    @pytest.fixture
    def service(self, make_service) -> LLMService:  # type: ignore[no-untyped-def]
        svc, _plugin = make_service(assistantModel="gpt-4")
        return svc

    def test_model_setting_is_read_from_profile(
        self, service: LLMService, mocker: MockerFixture, monkeypatch
    ) -> None:
        """For route_profile=PROFILE_CHAT, plugin.registryValue is called
        with PROFILES[PROFILE_CHAT].model_setting — not a hardcoded string.
        """
        from llm.profile import PROFILE_CHAT, PROFILES, Profile

        sentinel = Profile(
            id=PROFILE_CHAT,
            model_setting="SENTINEL_MODEL_KEY",
            prompt_id="chat",
            overlay_setting=None,
            max_output_tokens=None,
            force_search_on_explicit=False,
        )
        monkeypatch.setitem(PROFILES, PROFILE_CHAT, sentinel)

        mock_response = make_completion_response("ok")

        mocker.patch("llm.service.provider.chat", return_value=mock_response)

        # Capture every registryValue call so we can prove our sentinel
        # keys were the ones used.
        registry_calls: list[str] = []
        original_registry_value = service.plugin.registryValue

        def spy(key, *args, **kwargs):
            registry_calls.append(key)
            # Fall through to the real mock so existing test fixtures still
            # control timeouts, maxSteps, etc.
            return original_registry_value(key, *args, **kwargs)

        mocker.patch.object(service.plugin, "registryValue", side_effect=spy)

        service.assistant_completion(
            prompt="hello",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            route_profile=PROFILE_CHAT,
        )

        # The key is no longer a registry read at all — it comes from the
        # model's provider variable — so only the model setting is asserted.
        assert "SENTINEL_MODEL_KEY" in registry_calls
        assert "assistantModel" not in registry_calls

    def test_model_override_still_wins(
        self, service: LLMService, mocker: MockerFixture, monkeypatch
    ) -> None:
        """An explicit model_override= bypasses Profile.model_setting,
        matching the pre-refactor contract.
        """
        from llm.profile import PROFILE_CHAT, PROFILES, Profile

        sentinel = Profile(
            id=PROFILE_CHAT,
            model_setting="SENTINEL_MODEL_KEY",
            prompt_id="chat",
            overlay_setting=None,
            max_output_tokens=None,
            force_search_on_explicit=False,
        )
        monkeypatch.setitem(PROFILES, PROFILE_CHAT, sentinel)

        captured_model: list[str] = []

        mock_response = make_completion_response("ok")

        def capture(**kwargs):
            captured_model.append(kwargs.get("model"))
            return mock_response

        mocker.patch("llm.service.provider.chat", side_effect=capture)

        service.assistant_completion(
            prompt="hello",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            route_profile=PROFILE_CHAT,
            model_override="explicit-override-model",
        )

        # The override beat the Profile.model_setting lookup.
        assert captured_model == ["explicit-override-model"]


class TestOverlayReadsViaProfiles:
    """Plugin caller sites read the overlay key via PROFILES, not hardcoded.

    Swap a PROFILES entry with a sentinel ``overlay_setting`` and assert the
    sentinel key flows to ``plugin.registryValue`` at the call sites that
    drive chat-loop dispatch:

    - ``plugin.py`` structured-reminder fire (PROFILE_REMIND_ACTION)
    - ``plugin.py`` ``_ask_impl`` (PROFILE_CHAT or PROFILE_VERSE via
      ``effective_profile``)
    - ``service.py`` scheduled-task fire (PROFILE_REMIND_ACTION)

    These tests pin the Task 3 migration contract: a future regression
    that hardcodes ``"assistantSystemPrompt"`` at any of those sites
    would break here even though the runtime string is the same today.
    """

    def test_remind_action_fire_reads_overlay_via_profile(
        self, mock_irc: MagicMock, mocker: MockerFixture, monkeypatch
    ) -> None:
        """plugin.py structured-reminder fire reads
        ``PROFILES[PROFILE_REMIND_ACTION].overlay_setting``, not the
        hardcoded ``"assistantSystemPrompt"``.
        """
        from llm.profile import PROFILE_REMIND_ACTION, PROFILES, Profile
        from llm.service import AssistantResult, Spend

        sentinel = Profile(
            id=PROFILE_REMIND_ACTION,
            model_setting="assistantModel",
            prompt_id="remind_action",
            overlay_setting="SENTINEL_REMIND_OVERLAY",
            max_output_tokens=400,
            force_search_on_explicit=True,
        )
        monkeypatch.setitem(PROFILES, PROFILE_REMIND_ACTION, sentinel)

        # Capture registryValue keys while still serving defaults so the
        # rest of the fire path (rate limits, history, etc.) works.
        registry_calls: list[str] = []

        def spy(key, *args, **kwargs):
            registry_calls.append(key)
            # Return a sentinel-overlay-aware value: when the migrated
            # key is queried, return a marker so we can also verify the
            # value reached effective_prompt assembly downstream.
            if key == "SENTINEL_REMIND_OVERLAY":
                return "SENTINEL_OVERLAY_VALUE"
            return make_registry_side_effect()(key, *args, **kwargs)

        mocker.patch.object(LLM, "registryValue", side_effect=spy)
        plugin_init_patches(mocker)
        plugin = LLM(mock_irc)
        plugin.llm_service.sanitize_output.side_effect = lambda x: x

        mock_world = mocker.patch("llm.plugin.world")
        active_irc = mocker.MagicMock()
        active_irc.nick = "testbot"
        mock_world.ircs = [active_irc]

        mocker.patch.object(plugin, "_check_rate_limit", return_value=False)
        mocker.patch.object(plugin, "_gather_history", return_value=([], []))
        mocker.patch.object(plugin, "_get_user_memories", return_value=[])
        plugin.db.get_instruction.return_value = ""
        plugin.llm_service.assistant_completion.return_value = AssistantResult(
            content="done", spend=Spend.ZERO
        )

        event_name = "llm_remind_action_sentinel"
        row = make_reminder_row(
            event_name=event_name,
            nick="alice",
            channel="#ops",
            message="m",
            action_prompt="do x",
            account=None,
        )
        plugin._fire_reminder(row, 1_800_000_000.0)

        # Sentinel key was looked up; hardcoded key was NOT.
        assert "SENTINEL_REMIND_OVERLAY" in registry_calls
        assert "assistantSystemPrompt" not in registry_calls
        # And the sentinel-keyed value actually reached the system prompt.
        plugin.llm_service.assistant_completion.assert_called_once()
        sys_prompt = plugin.llm_service.assistant_completion.call_args.kwargs["system_prompt"]
        assert sys_prompt == "SENTINEL_OVERLAY_VALUE"

    def test_ask_path_overlay_uses_effective_profile(
        self, plugin_env, mocker: MockerFixture, monkeypatch
    ) -> None:
        """plugin.py _ask_impl reads ``PROFILES[effective_profile].overlay_setting``.

        When ``profile_override=PROFILE_VERSE`` is passed, the verse
        profile's sentinel overlay key flows to ``registryValue`` — proving
        the read is dispatched on ``effective_profile``, not hardcoded.
        """
        from llm.profile import PROFILE_VERSE, PROFILES, Profile
        from llm.service import AssistantResult, Spend

        plugin, mock_irc, mock_msg = plugin_env

        sentinel_verse = Profile(
            id=PROFILE_VERSE,
            model_setting="assistantModel",
            prompt_id="verse",
            overlay_setting="SENTINEL_VERSE_OVERLAY",
            max_output_tokens=None,
            force_search_on_explicit=False,
        )
        monkeypatch.setitem(PROFILES, PROFILE_VERSE, sentinel_verse)

        registry_calls: list[str] = []
        default_lookup = make_registry_side_effect()

        def spy(key, *args, **kwargs):
            registry_calls.append(key)
            if key == "SENTINEL_VERSE_OVERLAY":
                return "SENTINEL_VERSE_VALUE"
            return default_lookup(key, *args, **kwargs)

        plugin.registryValue.side_effect = spy

        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.assistant_completion.side_effect = None
        plugin.llm_service.assistant_completion.return_value = AssistantResult(
            content="ok", spend=Spend.ZERO
        )

        # Build a PreflightResult inline. Match the shape used by other
        # tests that drive _ask_impl through plugin.ask().
        from llm.plugin import PreflightResult

        pf = PreflightResult(blocked=False, caller=Caller(Identity("testnick", None), "#test"))

        plugin._ask_impl(
            mock_irc,
            mock_msg,
            "hello",
            pf,
            entry_route="verse",
            system_prompt_override="SCENE_CONTEXT",
            profile_override=PROFILE_VERSE,
        )

        assert "SENTINEL_VERSE_OVERLAY" in registry_calls
        # The hardcoded key must not have been read at the migrated site.
        # (It may still appear elsewhere — e.g. nothing else in _ask_impl
        # reads it — so a plain absence assertion is the right shape.)
        assert "assistantSystemPrompt" not in registry_calls

    def test_scheduled_task_fire_reads_overlay_via_profile(
        self, make_service, mocker: MockerFixture, monkeypatch
    ) -> None:
        """service.py scheduled-task fire reads
        ``PROFILES[PROFILE_REMIND_ACTION].overlay_setting``.
        """
        from llm.persistence import ScheduledLlmTaskRow
        from llm.profile import PROFILE_REMIND_ACTION, PROFILES, Profile

        sentinel = Profile(
            id=PROFILE_REMIND_ACTION,
            model_setting="assistantModel",
            prompt_id="remind_action",
            overlay_setting="SENTINEL_SCHED_OVERLAY",
            max_output_tokens=400,
            force_search_on_explicit=True,
        )
        monkeypatch.setitem(PROFILES, PROFILE_REMIND_ACTION, sentinel)

        service, plugin = make_service()

        registry_calls: list[str] = []
        default_lookup = make_registry_side_effect()

        def spy(key, *args, **kwargs):
            registry_calls.append(key)
            if key == "SENTINEL_SCHED_OVERLAY":
                return "SENTINEL_SCHED_VALUE"
            if key == "bridgeScheduledTaskLimit":
                return 5
            return default_lookup(key, *args, **kwargs)

        plugin.registryValue.side_effect = spy

        # Stub plugin internals the dispatch path touches. The fire plumbing
        # lives in plugin._run_unattended_assistant (shared with reminder
        # action fires); bind the real helpers to the mock plugin so the
        # dispatch path exercises the real overlay read.
        plugin._check_rate_limit.return_value = False
        plugin._gather_history.return_value = ([], [])
        plugin._get_user_memories.return_value = []
        plugin.db.get_instruction.return_value = ""
        plugin._pending_task_fns.return_value = {}
        plugin.llm_service = service
        plugin._unattended_ask_rate_limited = LLM._unattended_ask_rate_limited.__get__(plugin)
        plugin._run_unattended_assistant = LLM._run_unattended_assistant.__get__(plugin)
        mocker.patch("llm.service.ircdb.checkCapability", return_value=True)
        mocker.patch.object(
            service,
            "assistant_completion",
            return_value=mocker.MagicMock(
                content="ok",
                model="m",
                prompt_tokens=0,
                completion_tokens=0,
                cost=0.0,
                error=None,
            ),
        )

        # Drive the scheduled-task dispatch directly with a minimal row.
        irc = mocker.MagicMock()
        irc.network = "afternet"
        msg = mocker.MagicMock()
        msg.tagged.return_value = None
        fake_world = mocker.patch("llm.service.world", autospec=False, create=True)
        fake_world.getIrc.return_value = irc
        fake_world.ircs = [irc]

        row = ScheduledLlmTaskRow(
            id=1,
            event_name="scheduled_llm_task_1",
            creator_nick="rdrake",
            account="rdrake_a",
            channel="#t",
            network="afternet",
            wire_msg=":rdrake!u@h PRIVMSG #t :@ask do x",
            prompt="do x",
            fire_at=0.0,
            created_at=0.0,
            recurrence_seconds=None,
            recurrence_rrule=None,
            chain_position=1,
            watch_mode=False,
            reply_target=None,
        )

        service._dispatch_scheduled_task(irc, msg, row)

        assert "SENTINEL_SCHED_OVERLAY" in registry_calls
        assert "assistantSystemPrompt" not in registry_calls
