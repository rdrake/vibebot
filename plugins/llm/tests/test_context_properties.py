"""Property-based state-machine tests for ConversationContext.

Subsumes the per-user / per-channel isolation, case-insensitivity, and
trim-to-max example tests in ``test_context.py``.
"""

from __future__ import annotations

from hypothesis import HealthCheck, settings
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule
from hypothesis.strategies import sampled_from, text
from llm.context import ContextConfig, ConversationContext

NICKS = ["alice", "Alice", "BOB", "charlie", "dave"]
CHANNELS = ["#a", "#b", "#priv1"]
ROLES = ["user", "assistant"]


class ConversationContextMachine(RuleBasedStateMachine):
    """State machine exercising ConversationContext invariants.

    Both caps are deliberately small so the trim invariants are exercised.
    ``enabled=True`` is fixed: when disabled, ``get_messages`` returns ``[]``
    unconditionally and the case-insensitivity/isolation invariants
    collapse to vacuous truths.
    """

    def __init__(self) -> None:
        super().__init__()
        self.cfg = ContextConfig(
            max_messages=4,
            timeout_minutes=30,
            enabled=True,
            channel_max_messages=3,
        )
        self.ctx = ConversationContext(self.cfg)
        # Personal shadow model: (nick.lower(), channel.lower()) -> [(role, content)]
        self.personal: dict[tuple[str, str], list[tuple[str, str]]] = {}
        # Channel shadow model: channel.lower() -> [(nick, role, content)]
        self.channel: dict[str, list[tuple[str, str, str]]] = {}

    @rule(
        nick=sampled_from(NICKS),
        channel=sampled_from(CHANNELS),
        role=sampled_from(ROLES),
        content=text(max_size=200),
    )
    def add_message(self, nick: str, channel: str, role: str, content: str) -> None:
        self.ctx.add_message(nick, channel, role, content)
        key = (nick.lower(), channel.lower())
        msgs = self.personal.setdefault(key, [])
        msgs.append((role, content))
        if len(msgs) > self.cfg.max_messages:
            self.personal[key] = msgs[-self.cfg.max_messages :]

    @rule(
        channel=sampled_from(CHANNELS),
        nick=sampled_from(NICKS),
        role=sampled_from(ROLES),
        content=text(max_size=200),
    )
    def add_channel_message(self, channel: str, nick: str, role: str, content: str) -> None:
        self.ctx.add_channel_message(channel, nick, role, content)
        ch_key = channel.lower()
        msgs = self.channel.setdefault(ch_key, [])
        msgs.append((nick, role, content))
        if len(msgs) > self.cfg.channel_max_messages:
            self.channel[ch_key] = msgs[-self.cfg.channel_max_messages :]

    @rule(nick=sampled_from(NICKS), channel=sampled_from(CHANNELS))
    def clear(self, nick: str, channel: str) -> None:
        self.ctx.clear(nick, channel)
        self.personal.pop((nick.lower(), channel.lower()), None)
        # Postcondition: immediately after clear, the conversation is empty.
        assert self.ctx.get_messages(nick, channel) == []

    @rule(channel=sampled_from(CHANNELS))
    def clear_channel(self, channel: str) -> None:
        self.ctx.clear_channel(channel)
        self.channel.pop(channel.lower(), None)

    @rule()
    def clear_all(self) -> None:
        self.ctx.clear_all()
        self.personal.clear()
        self.channel.clear()

    # ---- invariants ----

    @invariant()
    def personal_trim(self) -> None:
        for nick_l, chan_l in self.personal:
            assert len(self.ctx.get_messages(nick_l, chan_l)) <= self.cfg.max_messages

    @invariant()
    def channel_trim(self) -> None:
        for chan_l in self.channel:
            assert len(self.ctx.get_channel_messages(chan_l)) <= self.cfg.channel_max_messages

    @invariant()
    def case_insensitive_personal_lookup(self) -> None:
        for nick_l, chan_l in self.personal:
            upper = self.ctx.get_messages(nick_l.upper(), chan_l.upper())
            lower = self.ctx.get_messages(nick_l.lower(), chan_l.lower())
            assert upper == lower

    @invariant()
    def isolation(self) -> None:
        # Distinct (nick.lower(), channel.lower()) keys never share content.
        keys = list(self.personal.keys())
        for i, key_a in enumerate(keys):
            msgs_a = self.ctx.get_messages(*key_a)
            shadow_a = [{"role": r, "content": c} for (r, c) in self.personal[key_a]]
            assert msgs_a == shadow_a
            for key_b in keys[i + 1 :]:
                if key_a == key_b:
                    continue
                msgs_b = self.ctx.get_messages(*key_b)
                shadow_b = [{"role": r, "content": c} for (r, c) in self.personal[key_b]]
                # If both shadows match the actual returned lists, isolation holds.
                # Re-assert against shadow_b as well (b's content is what b expects).
                assert msgs_b == shadow_b

    @invariant()
    def returned_messages_are_deep_copied(self) -> None:
        # Mutating a returned dict's value must not affect internal state.
        for nick_l, chan_l in self.personal:
            result = self.ctx.get_messages(nick_l, chan_l)
            if not result:
                continue
            original_role = result[0]["role"]
            result[0]["role"] = "MUTATED"
            again = self.ctx.get_messages(nick_l, chan_l)
            assert again[0]["role"] == original_role


ConversationContextMachine.TestCase.settings = settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
)
TestConversationContextStateMachine = ConversationContextMachine.TestCase
