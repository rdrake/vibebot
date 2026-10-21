"""Invariants for the consolidated profile module.

These tests pin the shape of ``llm.profile`` so future refactors don't
silently drop a profile key, mis-name a registry setting, or break the
behavior-preservation contract with the pre-refactor scattered data.
"""

from __future__ import annotations

import pytest
from llm import profile, prompts


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
