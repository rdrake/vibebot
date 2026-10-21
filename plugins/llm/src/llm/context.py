"""Conversation context management for LLM commands."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .persistence import LLMDatabase

# Minimum interval between full prune sweeps (seconds).
_PRUNE_INTERVAL = 30.0


class Role:
    """Message role constants for LLM conversations."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class ContextConfig:
    """Configuration for conversation context.

    Used both as default config in the ConversationContext constructor
    and as per-channel overrides passed to individual operations.
    """

    max_messages: int  # Maximum messages to keep per conversation
    timeout_minutes: int  # Clear context after this many minutes of inactivity
    enabled: bool = True  # Whether context is enabled
    channel_max_messages: int = 10  # Maximum messages in shared channel context


@dataclass
class Conversation:
    """A conversation's state.

    For personal context: messages have {"role": str, "content": str}
    For channel context: messages have {"nick": str, "role": str, "content": str}
    """

    messages: list[dict[str, str]] = field(default_factory=list)
    last_activity: float = field(default_factory=time.time)


class ConversationContext:
    """Thread-safe conversation context manager.

    Stores conversation history per user per channel with automatic expiry.

    All public methods accept an optional ``config`` parameter.  When provided
    it overrides the instance-level default for that single call, enabling
    per-channel configuration without having to mutate shared state.
    """

    def __init__(self, config: ContextConfig, *, db: LLMDatabase | None = None) -> None:
        """Initialize context manager.

        Args:
            config: Default context configuration (used when callers
                do not supply a per-call override)
            db: Optional database for persistence (None = in-memory only)
        """
        self.config = config
        self._db = db
        self._log = logging.getLogger("supybot.plugins.LLM")
        self._lock = Lock()
        # Key: (nick, channel) -> Conversation
        self._conversations: dict[tuple[str, str], Conversation] = {}
        # Key: channel -> Conversation (shared across all users)
        self._channel_contexts: dict[str, Conversation] = {}
        # Timestamp of last prune sweep (throttled to _PRUNE_INTERVAL)
        self._last_prune: float = 0.0

        if self._db is not None:
            self._load_from_db()

    def _best_effort_db(self, what: str, method: str, *args: object) -> None:
        """Run a persistence op by name, logging and swallowing any failure.

        Conversation context is primarily in-memory; the database is a
        best-effort persistence/recovery layer. A failed write or cleanup must
        never crash the caller (an IRC command worker) or plugin startup — the
        in-memory state stays the live source of truth for the session.
        """
        if self._db is None:
            return
        try:
            getattr(self._db, method)(*args)
        except Exception:
            self._log.warning("context db %s failed (continuing)", what, exc_info=True)

    def _load_from_db(self) -> None:
        """Load persisted conversations from the database at startup."""
        assert self._db is not None
        log = logging.getLogger("supybot.plugins.LLM")
        rows = self._db.load_conversations()
        timeout_seconds = self.config.timeout_minutes * 60
        now = time.time()
        loaded = 0
        for nick, channel, messages, last_activity in rows:
            if now - last_activity > timeout_seconds:
                self._best_effort_db("startup delete", "delete_conversation", nick, channel)
                continue
            key = (nick, channel)  # already lowercased by load_conversations
            self._conversations[key] = Conversation(messages=messages, last_activity=last_activity)
            loaded += 1
        if loaded:
            log.info("Loaded %i conversation(s) from database", loaded)

    def update_config(self, config: ContextConfig) -> None:
        """Update default configuration without losing conversation data.

        Args:
            config: New configuration to apply
        """
        with self._lock:
            self.config = config

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        return (
            f"ConversationContext(conversations={len(self._conversations)}, "
            f"channels={len(self._channel_contexts)}, enabled={self.config.enabled})"
        )

    def _get_key(self, nick: str, channel: str) -> tuple[str, str]:
        """Generate conversation key.

        Args:
            nick: User's IRC nick
            channel: IRC channel

        Returns:
            Tuple key for the conversation
        """
        return (nick.lower(), channel.lower())

    def _effective(self, config: ContextConfig | None) -> ContextConfig:
        """Return the per-call config or fall back to the instance default."""
        return config if config is not None else self.config

    def _is_expired(self, conversation: Conversation, cfg: ContextConfig) -> bool:
        """Check if a conversation has expired.

        Args:
            conversation: Conversation to check
            cfg: Configuration to use for timeout/enabled checks

        Returns:
            True if expired
        """
        if not cfg.enabled:
            return True
        timeout_seconds = cfg.timeout_minutes * 60
        return time.time() - conversation.last_activity > timeout_seconds

    @staticmethod
    def _is_stale(conversation: Conversation, max_age_seconds: int | None) -> bool:
        """True when last activity is older than ``max_age_seconds``.

        Returns False when ``max_age_seconds`` is None (no freshness gate).
        """
        if max_age_seconds is None:
            return False
        return time.time() - conversation.last_activity > max_age_seconds

    def _prune_expired(self, cfg: ContextConfig, *, force: bool = False) -> None:
        """Remove expired conversations. Must be called with lock held.

        Pruning is throttled to at most once per ``_PRUNE_INTERVAL`` seconds
        to avoid O(n) scans on every add/get call.  Pass ``force=True`` to
        bypass the throttle (used by ``get_stats``).
        """
        now = time.time()
        if not force and now - self._last_prune < _PRUNE_INTERVAL:
            return
        self._last_prune = now

        # Use instance default config for prune sweep to avoid
        # cross-channel config mismatch (see design doc).
        prune_cfg = self.config
        expired_keys = [
            key for key, conv in self._conversations.items() if self._is_expired(conv, prune_cfg)
        ]
        for key in expired_keys:
            del self._conversations[key]
            self._best_effort_db("prune delete", "delete_conversation", key[0], key[1])

        # Also prune expired channel contexts
        expired_channels = [
            ch for ch, ctx in self._channel_contexts.items() if self._is_expired(ctx, prune_cfg)
        ]
        for ch in expired_channels:
            del self._channel_contexts[ch]

    def add_message(
        self,
        nick: str,
        channel: str,
        role: str,
        content: str,
        *,
        config: ContextConfig | None = None,
        persist: bool = True,
    ) -> None:
        """Add a message to conversation history.

        Args:
            nick: User's IRC nick
            channel: IRC channel
            role: Message role ("user" or "assistant")
            content: Message content
            config: Per-channel config override (uses instance default if None)
            persist: Whether to persist to database (default True)
        """
        with self._lock:
            cfg = self._effective(config)
            if not cfg.enabled:
                return

            self._prune_expired(cfg)

            key = self._get_key(nick, channel)
            if key not in self._conversations:
                self._conversations[key] = Conversation()

            conv = self._conversations[key]
            conv.messages.append({"role": role, "content": content})
            conv.last_activity = time.time()

            # Trim to max messages, keeping most recent
            if len(conv.messages) > cfg.max_messages:
                # Remove oldest messages, keeping most recent
                conv.messages = conv.messages[-cfg.max_messages :]

            if persist:
                self._best_effort_db(
                    "save", "save_conversation", nick, channel, conv.messages, conv.last_activity
                )

    def get_messages(
        self,
        nick: str,
        channel: str,
        *,
        config: ContextConfig | None = None,
        max_age_seconds: int | None = None,
    ) -> list[dict[str, str]]:
        """Get conversation history as chat messages.

        Args:
            nick: User's IRC nick
            channel: IRC channel
            config: Per-channel config override (uses instance default if None)
            max_age_seconds: If set, return [] when the conversation's
                last activity is older than this many seconds. Used by
                routes like draw that only want very fresh context.

        Returns:
            List of OpenAI-format message dicts
        """
        with self._lock:
            cfg = self._effective(config)
            if not cfg.enabled:
                return []

            self._prune_expired(cfg)

            key = self._get_key(nick, channel)
            conv = self._conversations.get(key)

            if conv is None or self._is_expired(conv, cfg):
                return []

            if self._is_stale(conv, max_age_seconds):
                return []

            # Return deep copies to prevent external modification
            return [dict(msg) for msg in conv.messages]

    def add_channel_message(
        self,
        channel: str,
        nick: str,
        role: str,
        content: str,
        *,
        config: ContextConfig | None = None,
    ) -> None:
        """Add a message to shared channel context.

        Channel context is shared across all users in a channel, allowing
        the bot to follow group conversations.

        Args:
            channel: IRC channel
            nick: Nick of who sent the message
            role: Message role ("user" or "assistant")
            content: Message content
            config: Per-channel config override (uses instance default if None)
        """
        with self._lock:
            cfg = self._effective(config)
            if not cfg.enabled:
                return

            self._prune_expired(cfg)

            ch_key = channel.lower()
            if ch_key not in self._channel_contexts:
                self._channel_contexts[ch_key] = Conversation()

            ctx = self._channel_contexts[ch_key]
            ctx.messages.append({"nick": nick, "role": role, "content": content})
            ctx.last_activity = time.time()

            # Trim to max messages
            if len(ctx.messages) > cfg.channel_max_messages:
                ctx.messages = ctx.messages[-cfg.channel_max_messages :]

    def get_channel_messages(
        self,
        channel: str,
        exclude_nick: str | None = None,
        *,
        config: ContextConfig | None = None,
        max_age_seconds: int | None = None,
    ) -> list[dict[str, str]]:
        """Get shared channel context.

        Args:
            channel: IRC channel
            exclude_nick: Optional nick to exclude (typically the current user,
                since their messages are in personal context)
            config: Per-channel config override (uses instance default if None)
            max_age_seconds: If set, return [] when the channel context's
                last activity is older than this many seconds.

        Returns:
            List of channel messages with nick, role, and content
        """
        with self._lock:
            cfg = self._effective(config)
            if not cfg.enabled:
                return []

            self._prune_expired(cfg)

            ch_key = channel.lower()
            ctx = self._channel_contexts.get(ch_key)

            if ctx is None or self._is_expired(ctx, cfg):
                return []

            if self._is_stale(ctx, max_age_seconds):
                return []

            # Filter out excluded nick if specified
            if exclude_nick:
                exclude_lower = exclude_nick.lower()
                return [dict(msg) for msg in ctx.messages if msg["nick"].lower() != exclude_lower]

            return [dict(msg) for msg in ctx.messages]

    def clear(self, nick: str, channel: str) -> bool:
        """Clear conversation context for a specific user.

        Args:
            nick: User's IRC nick
            channel: IRC channel

        Returns:
            True if context was cleared, False if none existed
        """
        with self._lock:
            key = self._get_key(nick, channel)
            if key in self._conversations:
                del self._conversations[key]
                self._best_effort_db("clear delete", "delete_conversation", nick, channel)
                return True
            return False

    def clear_channel(self, channel: str) -> bool:
        """Clear shared volatile context for a channel."""
        with self._lock:
            ch_key = channel.lower()
            if ch_key in self._channel_contexts:
                del self._channel_contexts[ch_key]
                return True
            return False

    def clear_all(self) -> int:
        """Clear all conversation and channel contexts.

        Returns:
            Number of conversations cleared (not including channel contexts)
        """
        with self._lock:
            count = len(self._conversations)
            self._conversations.clear()
            self._channel_contexts.clear()
            self._best_effort_db("clear all", "delete_all_conversations")
            return count

    def get_user_stats(
        self,
        nick: str,
        channel: str,
        *,
        config: ContextConfig | None = None,
    ) -> dict[str, Any]:
        """Get context statistics for a specific user in a channel.

        Args:
            nick: User's IRC nick
            channel: IRC channel
            config: Per-channel config override (uses instance default if None)

        Returns:
            Dictionary with message_count, max_messages, seconds_until_expiry,
            and enabled.  seconds_until_expiry is 0 if no active conversation.
        """
        with self._lock:
            cfg = self._effective(config)
            if not cfg.enabled:
                return {
                    "message_count": 0,
                    "max_messages": cfg.max_messages,
                    "seconds_until_expiry": 0,
                    "enabled": False,
                }

            key = self._get_key(nick, channel)
            conv = self._conversations.get(key)

            if conv is None or self._is_expired(conv, cfg):
                return {
                    "message_count": 0,
                    "max_messages": cfg.max_messages,
                    "seconds_until_expiry": 0,
                    "enabled": True,
                }

            timeout_seconds = cfg.timeout_minutes * 60
            elapsed = time.time() - conv.last_activity
            remaining = max(0, timeout_seconds - elapsed)

            return {
                "message_count": len(conv.messages),
                "max_messages": cfg.max_messages,
                "seconds_until_expiry": int(remaining),
                "enabled": True,
            }

    def get_stats(self) -> dict[str, Any]:
        """Get context manager statistics.

        Returns:
            Dictionary with current statistics
        """
        with self._lock:
            self._prune_expired(self.config, force=True)

            total_messages = sum(len(conv.messages) for conv in self._conversations.values())
            channel_messages = sum(len(ctx.messages) for ctx in self._channel_contexts.values())

            return {
                "active_conversations": len(self._conversations),
                "total_messages": total_messages,
                "max_messages_per_conv": self.config.max_messages,
                "timeout_minutes": self.config.timeout_minutes,
                "enabled": self.config.enabled,
                "active_channels": len(self._channel_contexts),
                "channel_messages": channel_messages,
                "channel_max_messages": self.config.channel_max_messages,
            }
