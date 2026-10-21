"""Invariants for the consolidated profile module.

These tests pin the shape of ``llm.profile`` so future refactors don't
silently drop a profile key, mis-name a registry setting, or break the
behavior-preservation contract with the pre-refactor scattered data.
"""

from __future__ import annotations

import dataclasses

import pytest
from llm import profile, prompts


class TestProfilesRegistry:
    """PROFILES is the single source of truth for per-mode dispatch."""

    def test_registry_has_all_expected_keys(self):
        """PROFILES exposes every chat-loop profile by name."""
        assert set(profile.PROFILES.keys()) == {
            "chat",
            "draw",
            "verse",
            "remind_action",
        }

    @pytest.mark.parametrize(
        "pid",
        ["chat", "draw", "verse", "remind_action"],
    )
    def test_id_matches_dict_key(self, pid):
        """Profile.id matches its dict key — no copy-paste mismatches."""
        assert profile.PROFILES[pid].id == pid


class TestProfileResolution:
    """Each Profile resolves to live registry settings and a real prompt."""

    @pytest.mark.parametrize(
        "pid",
        ["chat", "draw", "verse", "remind_action"],
    )
    def test_prompt_id_is_a_real_prompt(self, pid):
        """profile.prompt_id is a valid key in prompts.PROMPTS."""
        assert profile.PROFILES[pid].prompt_id in prompts.PROMPTS

    @pytest.mark.parametrize(
        "pid",
        ["chat", "draw", "verse", "remind_action"],
    )
    def test_model_setting_is_a_known_registry_key(self, pid):
        """profile.model_setting matches a registerChannelValue in config.py."""
        valid_model_keys = {"assistantModel", "imageModel", "searchModel"}
        assert profile.PROFILES[pid].model_setting in valid_model_keys

    @pytest.mark.parametrize(
        "pid",
        ["chat", "draw", "verse", "remind_action"],
    )
    def test_overlay_setting_is_known_or_none(self, pid):
        """profile.overlay_setting is None or a known overlay key."""
        s = profile.PROFILES[pid].overlay_setting
        assert s is None or s == "assistantSystemPrompt"


class TestProfileToolsAlignment:
    """Profile.id is a valid input to assistant.get_tools_for_profile."""

    @pytest.mark.parametrize(
        "pid",
        ["chat", "draw", "verse", "remind_action"],
    )
    def test_get_tools_returns_nonempty(self, pid):
        """Every profile id resolves to at least one model-visible tool."""
        from llm.assistant import get_tools_for_profile

        tools = get_tools_for_profile(pid)
        assert len(tools) > 0


class TestBehaviorPreservation:
    """Pin pre-refactor scattered data; updates must be explicit."""

    EXPECTED_MAX_TOKENS = {"chat": 4096, "verse": 3000, "remind_action": 4096}
    EXPECTED_FORCE_SEARCH = {"chat", "remind_action"}
    # Per-profile sampling overrides; absent → None (provider default). Verse
    # alone tunes sampling to dampen its non-reasoning quality-collapse spiral.
    EXPECTED_PROMPT_IDS = {
        "chat": "chat",
        "draw": "draw",
        "verse": "verse",
        "remind_action": "remind_action",
    }
    # PROFILE_DRAW: no channel-overridable overlay — the @draw planner
    # constructs system_prompt from user_instruction + the framework prompt
    # directly, never reading a registry key.
    # PROFILE_VERSE / PROFILE_CHAT / PROFILE_REMIND_ACTION: all read
    # 'assistantSystemPrompt'.
    EXPECTED_OVERLAY = {
        "chat": "assistantSystemPrompt",
        "draw": None,
        "verse": "assistantSystemPrompt",
        "remind_action": "assistantSystemPrompt",
    }
    # Every chat-loop profile's assistant_completion fallback is
    # assistantModel. verseModel is a caller-side override
    # passed via model_override= rather than read by assistant_completion.
    _PROFILE_IDS = ("chat", "draw", "verse", "remind_action")
    EXPECTED_MODEL = dict.fromkeys(_PROFILE_IDS, "assistantModel")

    @pytest.mark.parametrize(
        "pid",
        ["chat", "draw", "verse", "remind_action"],
    )
    def test_max_output_tokens(self, pid):
        assert profile.PROFILES[pid].max_output_tokens == self.EXPECTED_MAX_TOKENS.get(pid)

    @pytest.mark.parametrize(
        "pid",
        ["chat", "draw", "verse", "remind_action"],
    )
    def test_force_search(self, pid):
        assert profile.PROFILES[pid].force_search_on_explicit == (pid in self.EXPECTED_FORCE_SEARCH)

    @pytest.mark.parametrize(
        "pid",
        ["chat", "draw", "verse", "remind_action"],
    )
    def test_prompt_id(self, pid):
        assert profile.PROFILES[pid].prompt_id == self.EXPECTED_PROMPT_IDS[pid]

    @pytest.mark.parametrize(
        "pid",
        ["chat", "draw", "verse", "remind_action"],
    )
    def test_overlay_setting_value(self, pid):
        assert profile.PROFILES[pid].overlay_setting == self.EXPECTED_OVERLAY[pid]

    @pytest.mark.parametrize(
        "pid",
        ["chat", "draw", "verse", "remind_action"],
    )
    def test_model_setting_value(self, pid):
        assert profile.PROFILES[pid].model_setting == self.EXPECTED_MODEL[pid]


class TestProfileImmutability:
    """Profile is a frozen dataclass — attempts to mutate must raise."""

    def test_profile_is_frozen(self):
        """Assigning to a Profile field raises FrozenInstanceError."""
        p = profile.PROFILES["chat"]
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.id = "mutated"  # type: ignore[misc]
