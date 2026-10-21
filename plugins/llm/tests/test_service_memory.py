"""Service memory: injection, extraction, cleanup, IRC teaser."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from .conftest import make_completion_response

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


class TestSummarize:
    """Tests for summarize_for_irc() and the shared ask-completion path."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        """Set up test fixtures."""
        self.mocker = mocker
        self.make_service = make_service
        self.service, self.mock_plugin = make_service(
            assistantModel="gpt-4",
            timeout=30,
        )

    def test_summarize_returns_none_on_empty_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN an empty provider variable WHEN summarize_for_irc called THEN returns None."""
        self.service, self.mock_plugin = self.make_service(assistantModel="xai/grok-4.3")
        monkeypatch.setenv("XAI_API_KEY", "")
        mock_completion = self.mocker.patch("llm.service.provider.chat")

        result = self.service.summarize_for_irc("content")

        assert result is None
        mock_completion.assert_not_called()

    def test_summarize_returns_none_on_exception(self) -> None:
        """GIVEN API error WHEN summarize_for_irc called THEN returns None gracefully."""
        self.mocker.patch("llm.service.provider.chat", side_effect=Exception("API error"))
        result = self.service.summarize_for_irc("content")

        assert result is None

    def test_summarize_returns_none_on_empty_response(self) -> None:
        """GIVEN empty response WHEN summarize_for_irc called THEN returns None."""
        mock_response = make_completion_response("")

        self.mocker.patch("llm.service.provider.chat", return_value=mock_response)
        result = self.service.summarize_for_irc("content")

        assert result is None

    def test_summarize_uses_ask_model(self) -> None:
        """GIVEN summarize_for_irc call WHEN API called THEN uses the ask model."""
        completion_kwargs = {}

        def capture_kwargs(**kwargs):
            completion_kwargs.update(kwargs)
            return make_completion_response("Summary")

        self.mocker.patch("llm.service.provider.chat", side_effect=capture_kwargs)
        self.service.summarize_for_irc("content")

        assert completion_kwargs["model"] == "gpt-4"

    def test_summarize_uses_channel_for_model_lookup(self) -> None:
        """GIVEN channel WHEN summarize_for_irc called THEN passes channel for model config."""
        registry_calls = []

        def track_registry(key, channel=None):
            registry_calls.append((key, channel))
            return {"assistantModel": "gpt-4", "timeout": 30, "commandPrefixes": []}.get(key)

        self.mock_plugin.registryValue = self.mocker.Mock(side_effect=track_registry)

        mock_response = make_completion_response("Summary")

        self.mocker.patch("llm.service.provider.chat", return_value=mock_response)
        self.service.summarize_for_irc("content", channel="#test")

        # assistantModel should be called with channel
        model_call = next(c for c in registry_calls if c[0] == "assistantModel")
        assert model_call[1] == "#test"

    def test_summarize_for_irc_returns_one_line_teaser(self) -> None:
        """GIVEN content WHEN IRC teaser requested THEN returns one compact line."""
        mock_response = make_completion_response(
            "  Liberia's history spans colonization, independence,\n"
            "  coups, civil war, and recovery. Extra text that should be trimmed."
        )
        mock_completion = self.mocker.patch("llm.service.provider.chat", return_value=mock_response)

        result = self.service.summarize_for_irc("long answer", channel="#test", max_chars=72)

        assert result == "Liberia's history spans colonization, independence, coups, civil war,"
        messages = mock_completion.call_args.kwargs["messages"]
        assert "one sentence" in messages[0]["content"]
        assert "no Markdown" in messages[0]["content"]

    def test_summarize_for_irc_returns_none_on_missing_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GIVEN no key for the ask model's provider WHEN teaser requested THEN None."""
        self.service, self.mock_plugin = self.make_service(assistantModel="xai/grok-4.3")
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        mock_completion = self.mocker.patch("llm.service.provider.chat")

        result = self.service.summarize_for_irc("long answer", channel="#test", max_chars=80)

        assert result is None
        mock_completion.assert_not_called()


class TestMemoryInjection:
    """Test memory injection into system prompts."""

    def test_completion_with_memories_injects_into_prompt(
        self, make_service, mocker: MockerFixture
    ) -> None:
        """GIVEN memories WHEN completion called THEN facts in a user message after system+context, NOT in the system prompt (preserves prompt-cache stability)."""
        service, mock_plugin = make_service()
        mock_chat = mocker.patch("llm.service.provider.chat")
        mock_chat.return_value = make_completion_response("Hello!")
        service.completion("hi", command="ask", memories=["likes Python", "lives in Toronto"])
        call_args = mock_chat.call_args
        messages = call_args.kwargs["messages"]
        system_msg = next(m for m in messages if m["role"] == "system")
        assert "likes Python" not in system_msg["content"]
        assert "lives in Toronto" not in system_msg["content"]
        user_blob = "\n".join(
            m.get("content", "")
            for m in messages
            if m.get("role") == "user" and isinstance(m.get("content"), str)
        )
        assert "likes Python" in user_blob
        assert "lives in Toronto" in user_blob

    def test_memories_wrapped_in_data_delimiters(self, make_service, mocker: MockerFixture) -> None:
        """Memories are user-authored and persistent, so a poisoned fact must
        not pose as an instruction: they are fenced in <user_memory> markers
        the model is told to treat as data."""
        service, mock_plugin = make_service()
        mock_chat = mocker.patch("llm.service.provider.chat")
        mock_chat.return_value = make_completion_response("Hello!")
        service.completion("hi", command="ask", memories=["likes Python"])
        call_args = mock_chat.call_args
        messages = call_args.kwargs["messages"]
        mem_msg = next(
            m
            for m in messages
            if m["role"] == "user" and "likes Python" in str(m.get("content", ""))
        )
        assert "<user_memory>" in mem_msg["content"]
        assert "</user_memory>" in mem_msg["content"]

    def test_user_instruction_is_user_role_data_not_system(
        self, make_service, mocker: MockerFixture
    ) -> None:
        """A per-user instruction must not sit in the SYSTEM prompt (where it
        reads as developer authority). It rides in a user-role message fenced
        in <user_instruction> markers."""
        service, mock_plugin = make_service()
        mock_chat = mocker.patch("llm.service.provider.chat")
        mock_chat.return_value = make_completion_response("Bonjour!")
        service.completion("hi", command="ask", user_instruction="always answer in French")
        call_args = mock_chat.call_args
        messages = call_args.kwargs["messages"]
        system_msg = next(m for m in messages if m["role"] == "system")
        assert "always answer in French" not in system_msg["content"]
        instr_msg = next(
            m
            for m in messages
            if m["role"] == "user" and "always answer in French" in str(m.get("content", ""))
        )
        assert "<user_instruction>" in instr_msg["content"]
        assert "</user_instruction>" in instr_msg["content"]

    def test_completion_without_memories_no_section(
        self, make_service, mocker: MockerFixture
    ) -> None:
        """GIVEN no memories WHEN completion called THEN no memory section."""
        service, mock_plugin = make_service()
        mock_chat = mocker.patch("llm.service.provider.chat")
        mock_chat.return_value = make_completion_response("Hello!")
        service.completion("hi", command="ask")
        call_args = mock_chat.call_args
        messages = call_args.kwargs["messages"]
        system_msg = next(m for m in messages if m["role"] == "system")
        assert "What you know about this user" not in system_msg["content"]

    def test_completion_with_empty_memories_no_section(
        self, make_service, mocker: MockerFixture
    ) -> None:
        """GIVEN empty memories list WHEN completion called THEN no memory section."""
        service, mock_plugin = make_service()
        mock_chat = mocker.patch("llm.service.provider.chat")
        mock_chat.return_value = make_completion_response("Hello!")
        service.completion("hi", command="ask", memories=[])
        call_args = mock_chat.call_args
        messages = call_args.kwargs["messages"]
        system_msg = next(m for m in messages if m["role"] == "system")
        assert "What you know about this user" not in system_msg["content"]


class TestMemoryExtraction:
    """Test memory fact extraction from conversations."""

    def test_extract_memories_prompt_limits_facts(
        self, make_service, mocker: MockerFixture
    ) -> None:
        """GIVEN extraction prompt WHEN checked THEN contains strictness markers."""
        from llm.prompts import MEMORY_EXTRACTION_PROMPT

        assert "at most 2" in MEMORY_EXTRACTION_PROMPT.lower()
        assert "DO NOT SAVE" in MEMORY_EXTRACTION_PROMPT

    def test_extract_memories_returns_facts(self, make_service, mocker: MockerFixture) -> None:
        """GIVEN conversation with facts WHEN extracted THEN returns ExtractionResult."""
        service, mock_plugin = make_service()
        mock_chat = mocker.patch("llm.service.provider.chat")
        mock_response = make_completion_response('{"add": ["likes Python", "lives in Toronto"]}')
        mock_chat.return_value = mock_response
        result = service.extract_memories(
            "user1", "#test", "I love Python and live in Toronto", "Cool!", []
        )
        assert result.add == ["likes Python", "lives in Toronto"]

    def test_extract_memories_uses_memory_model(self, make_service, mocker: MockerFixture) -> None:
        """memoryModel, when set, replaces assistantModel for extraction only."""
        service, _ = make_service(
            assistantModel="xai/grok-4.3",
            memoryModel="xai/grok-4-fast",
        )
        mock_chat = mocker.patch("llm.service.provider.chat")
        mock_chat.return_value = make_completion_response('{"add": []}')
        service.extract_memories("user1", "#test", "hello", "hi", [])
        assert mock_chat.call_args.kwargs["model"] == "xai/grok-4-fast"

    def test_extract_memories_falls_back_to_assistant_model(
        self, make_service, mocker: MockerFixture
    ) -> None:
        service, _ = make_service(assistantModel="xai/grok-4.3", memoryModel="")
        mock_chat = mocker.patch("llm.service.provider.chat")
        mock_chat.return_value = make_completion_response('{"add": []}')
        service.extract_memories("user1", "#test", "hello", "hi", [])
        assert mock_chat.call_args.kwargs["model"] == "xai/grok-4.3"

    def test_extract_memories_empty_on_no_facts(self, make_service, mocker: MockerFixture) -> None:
        """GIVEN boring conversation WHEN extracted THEN returns empty result."""
        service, mock_plugin = make_service()
        mock_chat = mocker.patch("llm.service.provider.chat")
        mock_response = make_completion_response('{"add": []}')
        mock_chat.return_value = mock_response
        result = service.extract_memories("user1", "#test", "hello", "hi", [])
        assert result.add == []

    def test_extract_memories_empty_on_error(self, make_service, mocker: MockerFixture) -> None:
        """GIVEN API error WHEN extracting THEN returns empty result."""
        service, mock_plugin = make_service()
        mock_chat = mocker.patch("llm.service.provider.chat")
        mock_chat.side_effect = Exception("API down")
        result = service.extract_memories("user1", "#test", "hi", "hello", [])
        assert result.add == []

    def test_extract_memories_logs_and_records_error_on_exception(
        self, make_service, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        """GIVEN extraction raises THEN error field is populated AND traceback logged."""
        service, _ = make_service()
        mock_chat = mocker.patch("llm.service.provider.chat")
        mock_chat.side_effect = RuntimeError("boom")

        with caplog.at_level(logging.ERROR, logger="LLM"):
            result = service.extract_memories("user1", "#test", "hi", "hello", [])

        assert result.add == []
        assert result.error is not None
        assert any("extract_memories failed" in r.message for r in caplog.records)
        # The .exception() call records traceback info on the LogRecord
        assert any(r.exc_info is not None for r in caplog.records)

    def test_ask_completion_logs_a_warning_on_failure(
        self, make_service, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        """GIVEN _ask_completion raises THEN logs a WARNING (INFO never
        reaches the prod log) and returns None."""
        service, mock_plugin = make_service()
        mock_chat = mocker.patch("llm.service.provider.chat")
        mock_chat.side_effect = RuntimeError("nope")

        with caplog.at_level(logging.INFO, logger="LLM"):
            out = service._ask_completion("sys", "user", channel=None)

        assert out is None
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("Ask completion failed" in r.message for r in warnings)

    def test_extract_memories_empty_on_invalid_json(
        self, make_service, mocker: MockerFixture
    ) -> None:
        """GIVEN non-JSON response WHEN extracting THEN returns empty result."""
        service, mock_plugin = make_service()
        mock_chat = mocker.patch("llm.service.provider.chat")
        mock_response = make_completion_response("not json at all")
        mock_chat.return_value = mock_response
        result = service.extract_memories("user1", "#test", "hi", "hello", [])
        assert result.add == []

    def test_extract_memories_includes_existing_in_prompt(
        self, make_service, mocker: MockerFixture
    ) -> None:
        """GIVEN existing memories WHEN extracting THEN included in prompt."""
        service, mock_plugin = make_service()
        mock_chat = mocker.patch("llm.service.provider.chat")
        mock_response = make_completion_response('{"add": []}')
        mock_chat.return_value = mock_response
        service.extract_memories("user1", "#test", "hi", "hello", ["already knows Python"])
        call_args = mock_chat.call_args
        messages = call_args.kwargs["messages"]
        prompt_text = " ".join(m["content"] for m in messages)
        assert "already knows Python" in prompt_text

    def test_extract_memories_returns_reinforce_indices(
        self, make_service, mocker: MockerFixture
    ) -> None:
        """GIVEN candidates WHEN LLM reinforces THEN indices flow through."""
        service, _ = make_service()
        mock_chat = mocker.patch("llm.service.provider.chat")
        mock_response = make_completion_response('{"add": ["new fact"], "reinforce": [0, 2]}')
        mock_chat.return_value = mock_response
        result = service.extract_memories(
            "user1",
            "#test",
            "hi",
            "hello",
            [],
            existing_candidates=["a", "b", "c"],
        )
        assert result.add == ["new fact"]
        assert result.reinforce == [0, 2]

    def test_extract_memories_drops_out_of_range_reinforce(
        self, make_service, mocker: MockerFixture
    ) -> None:
        """GIVEN reinforce index >= candidate count WHEN parsed THEN dropped."""
        service, _ = make_service()
        mock_chat = mocker.patch("llm.service.provider.chat")
        mock_response = make_completion_response('{"add": [], "reinforce": [0, 5, -1, 1]}')
        mock_chat.return_value = mock_response
        result = service.extract_memories(
            "user1", "#test", "hi", "hello", [], existing_candidates=["a", "b"]
        )
        assert result.reinforce == [0, 1]

    def test_extract_memories_includes_candidates_in_prompt(
        self, make_service, mocker: MockerFixture
    ) -> None:
        """GIVEN candidate facts WHEN extracting THEN they appear indexed in prompt."""
        service, _ = make_service()
        mock_chat = mocker.patch("llm.service.provider.chat")
        mock_response = make_completion_response('{"add": [], "reinforce": []}')
        mock_chat.return_value = mock_response
        service.extract_memories(
            "user1",
            "#test",
            "hi",
            "hello",
            [],
            existing_candidates=["uses Arch Linux", "lives in Berlin"],
        )
        call_args = mock_chat.call_args
        messages = call_args.kwargs["messages"]
        prompt_text = " ".join(m["content"] for m in messages)
        assert "[0] uses Arch Linux" in prompt_text
        assert "[1] lives in Berlin" in prompt_text

    def test_extract_memories_system_prompt_is_byte_stable(
        self, make_service, mocker: MockerFixture
    ) -> None:
        """The system message must be byte-identical regardless of which user
        is being extracted, their memories, or pending candidates. xAI's
        prefix cache keys off these leading bytes — when per-user state
        leaks into the system role the cache resets every call and
        ``cached_tokens`` stays pinned at the provider's ~64-token baseline."""
        from llm.prompts import MEMORY_EXTRACTION_PROMPT

        service, _ = make_service()
        mock_chat = mocker.patch("llm.service.provider.chat")
        mock_response = make_completion_response('{"add": [], "reinforce": []}')
        mock_chat.return_value = mock_response

        service.extract_memories("alice", "#test", "hi", "hello", ["knows Python", "uses Arch"])
        first_system = mock_chat.call_args.kwargs["messages"][0]
        mock_chat.reset_mock()

        service.extract_memories(
            "bob",
            "#test",
            "yo",
            "hey",
            ["plays guitar"],
            existing_candidates=["likes coffee"],
        )
        second_system = mock_chat.call_args.kwargs["messages"][0]

        assert first_system == second_system
        assert first_system["content"] == MEMORY_EXTRACTION_PROMPT

    def test_extract_memories_user_message_carries_state(
        self, make_service, mocker: MockerFixture
    ) -> None:
        """Per-user state (known facts, pending candidates) must surface in
        the user role rather than the system prompt — that's what keeps the
        system prompt cache-stable while still feeding the model the context
        it needs to choose between add and reinforce."""
        service, _ = make_service()
        mock_chat = mocker.patch("llm.service.provider.chat")
        mock_response = make_completion_response('{"add": [], "reinforce": []}')
        mock_chat.return_value = mock_response

        service.extract_memories(
            "alice",
            "#test",
            "i moved to berlin",
            "nice",
            ["knows Python"],
            existing_candidates=["likes coffee"],
        )

        messages = mock_chat.call_args.kwargs["messages"]
        user_msg = messages[1]
        assert user_msg["role"] == "user"
        assert "knows Python" in user_msg["content"]
        assert "[0] likes coffee" in user_msg["content"]
        assert "i moved to berlin" in user_msg["content"]


class TestMemoryCleanup:
    """Test memory cleanup LLM call and validation."""

    def test_cleanup_returns_valid_edits(self, make_service, mocker: MockerFixture) -> None:
        """GIVEN memories with duplicates WHEN cleanup THEN returns drop/merge."""
        from llm.persistence import MemoryRow

        service, mock_plugin = make_service()
        mock_chat = mocker.patch("llm.service.provider.chat")
        mock_response = make_completion_response(
            '{"drop": [4], "merge": [{"indices": [1, 2], "text": "likes Python"}]}'
        )
        mock_chat.return_value = mock_response

        rows = [
            MemoryRow(10, "user1", "moved to Vancouver", "#test", 500.0),
            MemoryRow(11, "user1", "likes Python programming", "#test", 400.0),
            MemoryRow(12, "user1", "enjoys writing Python", "#test", 300.0),
            MemoryRow(13, "user1", "works at Acme", "#test", 200.0),
            MemoryRow(14, "user1", "asked about weather", "#test", 100.0),
        ]
        result = service.cleanup_memories("user1", "#test", rows)
        assert result.drop == [4]
        from llm.service import MergeOp

        assert result.merge == [MergeOp([1, 2], "likes Python")]

    def test_cleanup_returns_empty_on_error(self, make_service, mocker: MockerFixture) -> None:
        """GIVEN API error WHEN cleanup THEN returns empty result."""
        from llm.persistence import MemoryRow

        service, mock_plugin = make_service()
        mock_chat = mocker.patch("llm.service.provider.chat")
        mock_chat.side_effect = Exception("API down")

        rows = [
            MemoryRow(10, "user1", "fact a", "#test", 100.0),
            MemoryRow(11, "user1", "fact b", "#test", 200.0),
        ]
        result = service.cleanup_memories("user1", "#test", rows)
        assert result.drop == []
        assert result.merge == []
        assert result.error is not None

    def test_cleanup_rejects_invalid_json(self, make_service, mocker: MockerFixture) -> None:
        """GIVEN garbage LLM output WHEN cleanup THEN returns error result."""
        from llm.persistence import MemoryRow

        service, mock_plugin = make_service()
        mock_chat = mocker.patch("llm.service.provider.chat")
        mock_response = make_completion_response("not json at all")
        mock_chat.return_value = mock_response

        rows = [
            MemoryRow(10, "user1", "fact a", "#test", 100.0),
            MemoryRow(11, "user1", "fact b", "#test", 200.0),
        ]
        result = service.cleanup_memories("user1", "#test", rows)
        # Garbage output must fail in the json.loads except-branch, not in a
        # later structural validation branch.
        assert result.error is not None
        assert result.error.startswith("LLM call failed")
        assert result.drop == []
        assert result.merge == []

    def test_cleanup_rejects_duplicate_indices(self, make_service, mocker: MockerFixture) -> None:
        """GIVEN index in both drop and merge WHEN cleanup THEN returns error."""
        from llm.persistence import MemoryRow

        service, mock_plugin = make_service()
        mock_chat = mocker.patch("llm.service.provider.chat")
        mock_response = make_completion_response(
            '{"drop": [1], "merge": [{"indices": [0, 1], "text": "combined"}]}'
        )
        mock_chat.return_value = mock_response

        rows = [
            MemoryRow(10, "user1", "fact a", "#test", 100.0),
            MemoryRow(11, "user1", "fact b", "#test", 200.0),
            MemoryRow(12, "user1", "fact c", "#test", 300.0),
        ]
        result = service.cleanup_memories("user1", "#test", rows)
        assert result.error is not None

    def test_cleanup_rejects_out_of_range_index(self, make_service, mocker: MockerFixture) -> None:
        """GIVEN out-of-range index WHEN cleanup THEN returns error."""
        from llm.persistence import MemoryRow

        service, mock_plugin = make_service()
        mock_chat = mocker.patch("llm.service.provider.chat")
        mock_response = make_completion_response('{"drop": [5], "merge": []}')
        mock_chat.return_value = mock_response

        rows = [
            MemoryRow(10, "user1", "fact a", "#test", 100.0),
            MemoryRow(11, "user1", "fact b", "#test", 200.0),
        ]
        result = service.cleanup_memories("user1", "#test", rows)
        assert result.error is not None

    def test_cleanup_rejects_empty_merge_text(self, make_service, mocker: MockerFixture) -> None:
        """GIVEN merge with empty text WHEN cleanup THEN returns error."""
        from llm.persistence import MemoryRow

        service, mock_plugin = make_service()
        mock_chat = mocker.patch("llm.service.provider.chat")
        mock_response = make_completion_response(
            '{"drop": [], "merge": [{"indices": [0, 1], "text": ""}]}'
        )
        mock_chat.return_value = mock_response

        rows = [
            MemoryRow(10, "user1", "fact a", "#test", 100.0),
            MemoryRow(11, "user1", "fact b", "#test", 200.0),
        ]
        result = service.cleanup_memories("user1", "#test", rows)
        assert result.error is not None

    def test_cleanup_rejects_zero_surviving_memories(
        self, make_service, mocker: MockerFixture
    ) -> None:
        """GIVEN all memories dropped WHEN cleanup THEN returns error."""
        from llm.persistence import MemoryRow

        service, mock_plugin = make_service()
        mock_chat = mocker.patch("llm.service.provider.chat")
        mock_response = make_completion_response('{"drop": [0, 1], "merge": []}')
        mock_chat.return_value = mock_response

        rows = [
            MemoryRow(10, "user1", "fact a", "#test", 100.0),
            MemoryRow(11, "user1", "fact b", "#test", 200.0),
        ]
        result = service.cleanup_memories("user1", "#test", rows)
        # Must reject via the surviving-count guard specifically (not some other
        # validation branch): all indices dropped -> zero survivors.
        assert result.error == "Cleanup would leave user with zero memories"
        # And it must reject WITHOUT proposing edits — no drop/merge are applied
        # when the guard fires.
        assert result.drop == []
        assert result.merge == []

    def test_cleanup_prompt_includes_indexed_memories(
        self, make_service, mocker: MockerFixture
    ) -> None:
        """GIVEN memories WHEN cleanup called THEN prompt lists them with indices."""
        from llm.persistence import MemoryRow

        service, mock_plugin = make_service()
        mock_chat = mocker.patch("llm.service.provider.chat")
        mock_response = make_completion_response('{"drop": [], "merge": []}')
        mock_chat.return_value = mock_response

        rows = [
            MemoryRow(10, "user1", "likes Python", "#test", 200.0),
            MemoryRow(11, "user1", "works at Acme", "#test", 100.0),
        ]
        service.cleanup_memories("user1", "#test", rows)

        call_args = mock_chat.call_args
        messages = call_args.kwargs["messages"]
        prompt_text = " ".join(m["content"] for m in messages)
        assert "[0] likes Python" in prompt_text
        assert "[1] works at Acme" in prompt_text

    def test_cleanup_uses_assistant_model(self, make_service, mocker: MockerFixture) -> None:
        """GIVEN cleanup call WHEN LLM invoked THEN uses assistantModel."""
        from llm.persistence import MemoryRow

        service, mock_plugin = make_service()
        mock_chat = mocker.patch("llm.service.provider.chat")
        mock_response = make_completion_response('{"drop": [], "merge": []}')
        mock_chat.return_value = mock_response

        rows = [MemoryRow(10, "user1", "fact", "#test", 100.0)]
        service.cleanup_memories("user1", "#test", rows)

        call_kwargs = mock_chat.call_args.kwargs
        from .conftest import TEST_MODEL

        assert call_kwargs["model"] == TEST_MODEL

    def test_cleanup_uses_registry_timeout(self, make_service, mocker: MockerFixture) -> None:
        """GIVEN custom timeout WHEN cleanup runs THEN LLM call uses registry value."""
        from llm.persistence import MemoryRow

        service, mock_plugin = make_service(timeout=123)
        mock_chat = mocker.patch("llm.service.provider.chat")
        mock_response = make_completion_response('{"drop": [], "merge": []}')
        mock_chat.return_value = mock_response

        rows = [MemoryRow(10, "user1", "fact", "#test", 100.0)]
        service.cleanup_memories("user1", "#test", rows)

        call_kwargs = mock_chat.call_args.kwargs
        assert call_kwargs["timeout"] == 123

    def test_cleanup_result_has_no_keep_field(self) -> None:
        """GIVEN CleanupResult WHEN inspected THEN has no keep field."""
        from llm.service import CleanupResult

        assert "keep" not in CleanupResult._fields

    def test_cleanup_keeps_unmentioned_indices(self, make_service, mocker: MockerFixture) -> None:
        """GIVEN LLM omits some indices WHEN cleanup THEN unmentioned indices are kept."""
        from llm.persistence import MemoryRow

        service, mock_plugin = make_service()
        mock_chat = mocker.patch("llm.service.provider.chat")
        mock_response = make_completion_response('{"drop": [2], "merge": []}')
        mock_chat.return_value = mock_response

        rows = [
            MemoryRow(10, "user1", "fact a", "#test", 300.0),
            MemoryRow(11, "user1", "fact b", "#test", 200.0),
            MemoryRow(12, "user1", "trivial fact", "#test", 100.0),
        ]
        result = service.cleanup_memories("user1", "#test", rows)
        assert result.error is None
        assert result.drop == [2]

    def test_cleanup_prompt_uses_new_format(self) -> None:
        """GIVEN cleanup prompt WHEN checked THEN uses new merge format without keep."""
        from llm.prompts import MEMORY_CLEANUP_PROMPT

        assert "keep" not in MEMORY_CLEANUP_PROMPT.lower()
        assert "Be aggressive" in MEMORY_CLEANUP_PROMPT


class TestCleanupMemoriesValidation:
    """Tests for cleanup_memories validation logic at lines 2694-2723."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        """Set up test fixtures."""
        self.mocker = mocker
        self.service, self.mock_plugin = make_service()

    def _mock_cleanup_response(self, parsed: object) -> None:
        """Mock provider.chat to return a JSON-encoded response."""
        import json

        response = make_completion_response(json.dumps(parsed))
        self.mocker.patch("llm.service.provider.chat", return_value=response)

    def _make_rows(self, count: int) -> list:
        """Create a list of MemoryRow objects for testing."""
        from llm.persistence import MemoryRow

        return [
            MemoryRow(id=i, nick="u", fact=f"fact{i}", source_channel="#c", created_at=0.0)
            for i in range(count)
        ]

    def test_not_a_dict(self) -> None:
        """GIVEN LLM returns a JSON string WHEN cleanup validates THEN error contains 'not a JSON object'."""
        self._mock_cleanup_response("just a string")
        result = self.service.cleanup_memories("u", "#c", self._make_rows(3))
        assert result.error is not None
        assert "not a JSON object" in result.error

    def test_drop_not_list(self) -> None:
        """GIVEN drop is not a list WHEN cleanup validates THEN error contains 'must be arrays'."""
        self._mock_cleanup_response({"drop": "x", "merge": []})
        result = self.service.cleanup_memories("u", "#c", self._make_rows(3))
        assert result.error is not None
        assert "must be arrays" in result.error

    def test_invalid_drop_index(self) -> None:
        """GIVEN drop index out of range WHEN cleanup validates THEN error contains 'Invalid drop index'."""
        self._mock_cleanup_response({"drop": [99], "merge": []})
        result = self.service.cleanup_memories("u", "#c", self._make_rows(3))
        assert result.error is not None
        assert "Invalid drop index" in result.error

    def test_bool_drop_index_is_rejected(self) -> None:
        """GIVEN drop [true] WHEN cleanup validates THEN it is an error, not memory #1.

        ``True`` is an ``int`` in Python; accepting it would delete index 1.
        """
        self._mock_cleanup_response({"drop": [True], "merge": []})
        result = self.service.cleanup_memories("u", "#c", self._make_rows(3))
        assert result.error is not None
        assert "Invalid drop index" in result.error

    def test_bool_merge_index_is_rejected(self) -> None:
        """GIVEN merge indices [false, true] WHEN cleanup validates THEN it is an error."""
        self._mock_cleanup_response(
            {"drop": [], "merge": [{"indices": [False, True], "text": "m"}]}
        )
        result = self.service.cleanup_memories("u", "#c", self._make_rows(3))
        assert result.error is not None
        assert "Merge index out of range" in result.error

    def test_non_dict_merge_entry(self) -> None:
        """GIVEN merge entry is a string WHEN cleanup validates THEN error contains 'Invalid merge entry'."""
        self._mock_cleanup_response({"drop": [], "merge": ["x"]})
        result = self.service.cleanup_memories("u", "#c", self._make_rows(3))
        assert result.error is not None
        assert "Invalid merge entry" in result.error

    def test_merge_with_zero_indices(self) -> None:
        """GIVEN merge entry with empty indices WHEN cleanup validates THEN error contains 'at least'."""
        self._mock_cleanup_response({"drop": [], "merge": [{"indices": [], "text": "merged"}]})
        result = self.service.cleanup_memories("u", "#c", self._make_rows(3))
        assert result.error is not None
        assert "at least" in result.error

    def test_merge_with_single_index(self) -> None:
        """GIVEN merge entry with one index WHEN cleanup validates THEN error
        contains 'at least' — a single-index merge is degenerate and the
        error message already promises at least 2."""
        self._mock_cleanup_response({"drop": [], "merge": [{"indices": [0], "text": "merged"}]})
        result = self.service.cleanup_memories("u", "#c", self._make_rows(3))
        assert result.error is not None
        assert "at least" in result.error

    def test_empty_merge_text(self) -> None:
        """GIVEN merge entry with empty text WHEN cleanup validates THEN error contains 'non-empty'."""
        self._mock_cleanup_response({"drop": [], "merge": [{"indices": [0, 1], "text": ""}]})
        result = self.service.cleanup_memories("u", "#c", self._make_rows(3))
        assert result.error is not None
        assert "non-empty" in result.error

    def test_duplicate_indices(self) -> None:
        """GIVEN index appears in both drop and merge WHEN cleanup validates THEN error contains 'Duplicate'."""
        self._mock_cleanup_response({"drop": [0], "merge": [{"indices": [0, 1], "text": "merged"}]})
        result = self.service.cleanup_memories("u", "#c", self._make_rows(3))
        assert result.error is not None
        assert "Duplicate" in result.error

    def test_merge_index_out_of_range(self) -> None:
        """GIVEN merge index exceeds memory count WHEN cleanup validates THEN error contains 'out of range'."""
        self._mock_cleanup_response({"drop": [], "merge": [{"indices": [0, 99], "text": "merged"}]})
        result = self.service.cleanup_memories("u", "#c", self._make_rows(3))
        assert result.error is not None
        assert "out of range" in result.error
