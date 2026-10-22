"""Stress tests for multi-user scenarios.

These tests verify the bot handles concurrent users, rapid requests,
and complex real-world usage patterns correctly.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any

import pytest
from llm.context import ContextConfig, ConversationContext
from llm.service import LLMService

from .conftest import FAKE_PROVIDER_KEYS

if TYPE_CHECKING:
    pass

pytestmark = pytest.mark.slow


class TestMultiUserContextIsolation:
    """Test context isolation under concurrent multi-user access."""

    @pytest.fixture
    def high_capacity_context(self) -> ConversationContext:
        """Create context with high capacity for stress testing."""
        config = ContextConfig(
            max_messages=100,
            timeout_minutes=60,
            channel_max_messages=200,
        )
        return ConversationContext(config)

    def test_many_users_simultaneous_add(self, high_capacity_context: ConversationContext) -> None:
        """GIVEN 50 users WHEN all add messages simultaneously THEN no data corruption."""
        ctx = high_capacity_context
        num_users = 50
        messages_per_user = 20
        errors: list[Exception] = []
        lock = threading.Lock()

        def user_activity(user_id: int) -> None:
            try:
                nick = f"user{user_id}"
                channel = "#stress"

                for i in range(messages_per_user):
                    # Add user message
                    ctx.add_message(nick, channel, "user", f"User {user_id} message {i}")
                    # Add assistant response
                    ctx.add_message(nick, channel, "assistant", f"Response to {user_id}:{i}")

                    # Occasionally read to create read/write contention
                    if i % 5 == 0:
                        messages = ctx.get_messages(nick, channel)
                        # Verify messages belong to this user
                        for msg in messages:
                            if "User" in msg["content"] and f"User {user_id}" not in msg["content"]:
                                with lock:
                                    errors.append(
                                        ValueError(f"User {user_id} saw foreign message: {msg}")
                                    )
            except Exception as e:
                with lock:
                    errors.append(e)

        # Run all users concurrently
        threads = []
        for i in range(num_users):
            t = threading.Thread(target=user_activity, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent access: {errors}"

        # Verify final state
        stats = ctx.get_stats()
        assert stats["active_conversations"] == num_users

    def test_channel_context_concurrent_multi_user(
        self, high_capacity_context: ConversationContext
    ) -> None:
        """GIVEN busy channel WHEN multiple users write simultaneously THEN correct ordering."""
        ctx = high_capacity_context
        num_users = 30
        messages_per_user = 10
        errors: list[Exception] = []
        lock = threading.Lock()

        def user_channel_activity(user_id: int) -> None:
            try:
                nick = f"user{user_id}"
                channel = "#busy"

                for i in range(messages_per_user):
                    content = f"[{user_id}:{i}] Message from user {user_id}"
                    ctx.add_channel_message(channel, nick, "user", content)

                    # Small delay to interleave with other users
                    time.sleep(0.001)

                    # Read channel history
                    messages = ctx.get_channel_messages(channel)

                    # Verify we can read messages
                    if len(messages) == 0:
                        with lock:
                            errors.append(ValueError(f"User {user_id} got empty channel history"))
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = []
        for i in range(num_users):
            t = threading.Thread(target=user_channel_activity, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"

        # Final channel should have messages (capped by max)
        messages = ctx.get_channel_messages("#busy")
        assert len(messages) > 0
        assert len(messages) <= 200  # channel_max_messages

    def test_rapid_context_clear_during_writes(
        self, high_capacity_context: ConversationContext
    ) -> None:
        """GIVEN ongoing writes WHEN context cleared THEN no race conditions."""
        ctx = high_capacity_context
        errors: list[Exception] = []
        lock = threading.Lock()
        stop_event = threading.Event()

        def writer(user_id: int) -> None:
            try:
                nick = f"user{user_id}"
                i = 0
                while not stop_event.is_set():
                    ctx.add_message(nick, "#test", "user", f"Message {i}")
                    i += 1
                    time.sleep(0.001)
            except Exception as e:
                with lock:
                    errors.append(e)

        def clearer() -> None:
            try:
                for _ in range(20):
                    time.sleep(0.01)
                    ctx.clear_all()
            except Exception as e:
                with lock:
                    errors.append(e)

        # Start writers
        writers = []
        for i in range(10):
            t = threading.Thread(target=writer, args=(i,))
            writers.append(t)
            t.start()

        # Start clearer
        clearer_thread = threading.Thread(target=clearer)
        clearer_thread.start()

        # Let it run
        clearer_thread.join()
        stop_event.set()

        for t in writers:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"


class TestConcurrentAPIKeyIsolation:
    """Test that API keys are isolated in concurrent requests."""

    def test_completion_key_isolation_across_providers(
        self, make_service, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GIVEN concurrent requests on different providers THEN no key bleeds.

        Keys are no longer per-request state, so the interesting contamination
        is now cross-provider: 20 threads interleaving two providers must each
        see their own model's variable, never a neighbour's.
        """
        models = ["xai/grok-4.3", "anthropic/claude-haiku-4-5"]
        expected = {
            "xai/grok-4.3": FAKE_PROVIDER_KEYS["XAI_API_KEY"],
            "anthropic/claude-haiku-4-5": FAKE_PROVIDER_KEYS["ANTHROPIC_API_KEY"],
        }
        observed: list[tuple[str, object]] = []
        lock = threading.Lock()

        def record(
            url: str, headers: dict[str, str], body: dict[str, Any], **kwargs: object
        ) -> dict:
            key = headers.get("x-api-key") or headers.get("Authorization", "").removeprefix(
                "Bearer "
            )
            with lock:
                observed.append((f"{kwargs['provider']}/{body['model']}", key))
            time.sleep(0.01)  # widen the interleaving window
            return {}

        # The key is read inside provider.chat, so stub the wire, not chat.
        monkeypatch.setattr("llm.provider._post", record)
        service, _ = make_service()

        def worker(index: int) -> None:
            model = models[index % len(models)]
            service._timed_completion("ask", model=model, messages=[], channel=None, timeout=5)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(observed) == 20
        assert len({key for _model, key in observed}) == 2
        for model, key in observed:
            assert key == expected[model], f"key bleed: {model} received another provider's key"


class TestOverlappingContextScenarios:
    """Test complex overlapping context scenarios."""

    @pytest.fixture
    def multi_channel_context(self) -> ConversationContext:
        """Create context for multi-channel testing."""
        config = ContextConfig(
            max_messages=50,
            timeout_minutes=30,
            channel_max_messages=100,
        )
        return ConversationContext(config)

    def test_user_active_in_multiple_channels(
        self, multi_channel_context: ConversationContext
    ) -> None:
        """GIVEN user in multiple channels WHEN chatting in all THEN proper isolation."""
        ctx = multi_channel_context
        nick = "activeuser"
        channels = ["#python", "#rust", "#general", "#help", "#offtopic"]
        messages_per_channel = 20

        errors = []
        lock = threading.Lock()

        def channel_activity(channel: str) -> None:
            try:
                for i in range(messages_per_channel):
                    ctx.add_message(nick, channel, "user", f"{channel} message {i}")
                    ctx.add_message(nick, channel, "assistant", f"Reply in {channel}")

                    # Read and verify
                    messages = ctx.get_messages(nick, channel)
                    for msg in messages:
                        if channel not in msg["content"] and "Reply" not in msg["content"]:
                            with lock:
                                errors.append(f"Wrong channel content in {channel}: {msg}")
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = []
        for channel in channels:
            t = threading.Thread(target=channel_activity, args=(channel,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"

        # Verify each channel has correct messages
        for channel in channels:
            messages = ctx.get_messages(nick, channel)
            assert len(messages) == messages_per_channel * 2

    def test_channel_history_exclude_with_concurrent_access(
        self, multi_channel_context: ConversationContext
    ) -> None:
        """GIVEN busy channel WHEN users read with exclude_nick THEN correct filtering."""
        ctx = multi_channel_context
        channel = "#busy"
        users = ["alice", "bob", "charlie", "dave", "eve"]
        errors = []
        lock = threading.Lock()

        # First, populate channel with messages
        for i in range(100):
            nick = users[i % len(users)]
            ctx.add_channel_message(channel, nick, "user", f"Message from {nick}: {i}")

        def read_with_exclude(user: str) -> None:
            try:
                for _ in range(50):
                    messages = ctx.get_channel_messages(channel, exclude_nick=user)
                    for msg in messages:
                        if msg["nick"].lower() == user.lower():
                            with lock:
                                errors.append(f"User {user} saw own message: {msg}")
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = []
        for user in users:
            t = threading.Thread(target=read_with_exclude, args=(user,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"


class TestLongRunningSessionScenarios:
    """Test scenarios involving long-running sessions and state management."""

    def test_context_expiry_under_load(self) -> None:
        """GIVEN very short timeout WHEN many users active THEN expiry works correctly."""
        config = ContextConfig(
            max_messages=100,
            timeout_minutes=0,  # Immediate expiry
        )
        ctx = ConversationContext(config)
        errors = []
        lock = threading.Lock()

        def user_with_expiry_check(user_id: int) -> None:
            try:
                nick = f"user{user_id}"
                channel = "#expiry"

                # Add messages
                for i in range(5):
                    ctx.add_message(nick, channel, "user", f"Message {i}")

                # Wait for expiry
                time.sleep(0.1)

                # Should be expired
                messages = ctx.get_messages(nick, channel)
                if len(messages) > 0:
                    with lock:
                        errors.append(f"User {user_id}: Messages didn't expire")
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = []
        for i in range(30):
            t = threading.Thread(target=user_with_expiry_check, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"

    def test_max_messages_eviction_under_load(self) -> None:
        """GIVEN low max_messages WHEN many messages added THEN proper FIFO eviction."""
        config = ContextConfig(
            max_messages=10,
            timeout_minutes=60,
        )
        ctx = ConversationContext(config)
        errors = []
        lock = threading.Lock()

        def user_with_eviction(user_id: int) -> None:
            try:
                nick = f"user{user_id}"
                channel = "#eviction"

                # Add more than max_messages
                for i in range(50):
                    ctx.add_message(nick, channel, "user", f"[{i}] Message")

                # Should only have last 10
                messages = ctx.get_messages(nick, channel)
                if len(messages) != 10:
                    with lock:
                        errors.append(f"User {user_id}: Expected 10, got {len(messages)}")

                # First message should be [40] (50 - 10)
                if "[40]" not in messages[0]["content"]:
                    with lock:
                        errors.append(
                            f"User {user_id}: Wrong first message: {messages[0]['content']}"
                        )
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = []
        for i in range(20):
            t = threading.Thread(target=user_with_eviction, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"


class TestHighVolumeFileOperations:
    """Test file operations under high volume."""

    @pytest.fixture
    def service(self, tmp_path, make_service) -> LLMService:
        """Create service with HTTP output config."""
        service, _ = make_service(
            httpRoot=str(tmp_path),
            httpUrlBase="http://localhost/llm",
            fileCleanupAge=24,
        )
        return service

    def test_concurrent_image_saving(self, service: LLMService, tmp_path) -> None:
        """GIVEN many concurrent image saves WHEN saving THEN all files created."""
        urls = []
        lock = threading.Lock()
        errors = []

        def save_image(image_id: int) -> None:
            try:
                # Create unique "image" data
                data = f"FAKE_IMAGE_DATA_{image_id}".encode()
                url = service._save_image_bytes(data, "png")
                if url:
                    with lock:
                        urls.append(url)
                else:
                    with lock:
                        errors.append(f"Failed to save image {image_id}")
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = []
        for i in range(50):
            t = threading.Thread(target=save_image, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(urls) == 50

        # Verify all files exist
        files = list(tmp_path.glob("*.png"))
        assert len(files) == 50
