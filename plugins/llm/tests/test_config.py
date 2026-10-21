"""Tests for LLM plugin configuration."""

from __future__ import annotations

import ast
import logging
import threading
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import supybot.registry as registry
from llm.config import ValidatedModelName

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

REPO_ROOT = Path(__file__).resolve().parents[3]
LLM_SRC = REPO_ROOT / "plugins" / "llm" / "src" / "llm"
REGISTER_METHODS = {"registerGlobalValue", "registerChannelValue"}


class TestValidatedModelName:
    """Test model name validation: a provider prefix we call, priced or warned."""

    def setup_method(self) -> None:
        """Clear warned-models cache between tests."""
        ValidatedModelName._warned.clear()

    @staticmethod
    def _make(default: str = "") -> ValidatedModelName:
        """Create a ValidatedModelName instance for testing."""
        return ValidatedModelName(default, "Test model config")

    def test_priced_model_accepted_silently(self, caplog: pytest.LogCaptureFixture) -> None:
        """GIVEN a priced model WHEN setValue called THEN accepts without a warning."""
        v = self._make()
        with caplog.at_level(logging.WARNING, logger="supybot.plugins.LLM.config"):
            v.setValue("xai/grok-4.3")
        assert v() == "xai/grok-4.3"
        assert caplog.text == ""

    def test_empty_string_accepted(self) -> None:
        """GIVEN empty string WHEN setValue called THEN accepts (not configured)."""
        v = self._make("xai/grok-4.3")
        v.setValue("")
        assert v() == ""

    def test_whitespace_stripped(self) -> None:
        """GIVEN model with surrounding whitespace WHEN setValue THEN strips it."""
        v = self._make()
        v.setValue("  xai/grok-4.3  ")
        assert v() == "xai/grok-4.3"

    def test_unpriced_model_accepted_with_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """GIVEN a provider we call but an unpriced model WHEN setValue THEN accepts with warning."""
        v = self._make()
        with caplog.at_level(logging.WARNING, logger="supybot.plugins.LLM.config"):
            v.setValue("anthropic/my-custom-model-xyz")
        assert v() == "anthropic/my-custom-model-xyz"
        assert "no price" in caplog.text.lower()

    @pytest.mark.parametrize(
        "model",
        ["totally-bogus-xyz", "gemini/gemini-2.0-flash", "openai/gpt-4", "gpt-4", "xai/"],
    )
    def test_unknown_provider_rejected(self, model: str) -> None:
        """GIVEN no provider prefix we call WHEN setValue THEN raises naming the valid ones."""
        v = self._make()
        with pytest.raises(registry.InvalidRegistryValue, match="Unknown model") as excinfo:
            v.setValue(model)
        assert "anthropic/" in str(excinfo.value)
        assert "xai/" in str(excinfo.value)

    def test_refusal_fallback_model_validates_without_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """GIVEN the priced refusal fallback WHEN setValue THEN no unpriced-model warning."""
        model = "xai/grok-4-1-fast-non-reasoning"
        v = self._make()
        with caplog.at_level(logging.WARNING, logger="supybot.plugins.LLM.config"):
            v.setValue(model)
        assert v() == model
        assert "no price" not in caplog.text.lower()

    def test_image_model_validates_without_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """GIVEN an image model priced per image WHEN setValue THEN no warning."""
        v = self._make()
        with caplog.at_level(logging.WARNING, logger="supybot.plugins.LLM.config"):
            v.setValue("xai/grok-imagine-image")
        assert v() == "xai/grok-imagine-image"
        assert "no price" not in caplog.text.lower()

    def test_warning_only_fires_once_per_model(self, caplog: pytest.LogCaptureFixture) -> None:
        """GIVEN unpriced model warned once WHEN setValue again THEN no duplicate warning."""
        v = self._make()
        with caplog.at_level(logging.WARNING, logger="supybot.plugins.LLM.config"):
            v.setValue("xai/my-dedup-test-model")
            caplog.clear()
            v.setValue("xai/my-dedup-test-model")
        assert caplog.text == ""


class TestConfigure:
    """Test plugin configuration wizard."""

    def test_configure_prints_setup_info(self, mocker: MockerFixture) -> None:
        """GIVEN configure function WHEN called THEN prints setup information."""
        from llm.config import configure

        output = StringIO()
        mocker.patch("sys.stdout", output)
        mocker.patch("supybot.conf.registerPlugin")
        configure(advanced=False)

        result = output.getvalue()
        assert "LLM Plugin Configuration" in result
        assert "API keys" in result
        assert "XAI_API_KEY" in result

    def test_configure_registers_plugin(self, mocker: MockerFixture) -> None:
        """GIVEN configure function WHEN called THEN registers plugin."""
        from llm.config import configure

        mocker.patch("sys.stdout", StringIO())
        mock_register = mocker.patch("supybot.conf.registerPlugin")
        configure(advanced=True)

        mock_register.assert_called_with("LLM", True)


class TestConfigValues:
    """Test configuration value registration."""

    def test_llm_plugin_registered(self) -> None:
        """GIVEN config module WHEN imported THEN LLM plugin is registered."""
        from llm import config

        assert hasattr(config, "LLM")

    def test_database_path_registered(self) -> None:
        """GIVEN LLM config WHEN checking databasePath THEN exists with empty default."""
        from llm import config  # noqa: F811

        value = config.LLM.databasePath()
        assert value == ""

    def test_long_reply_hybrid_defaults_registered(self) -> None:
        """GIVEN LLM config WHEN checking long reply UX THEN defaults are registered."""
        import supybot.conf as conf
        from llm import config  # noqa: F401

        assert conf.supybot.plugins.LLM.longReplyTeaserMaxChars() == 220

    def test_bridge_registry_values_registered_with_safe_defaults(self) -> None:
        """B1: bridgeEnabled defaults to False and bridgeAllowedPlugins to []."""
        import llm.config  # noqa: F401 — import side effect registers the values
        import supybot.conf as conf

        assert conf.supybot.plugins.LLM.bridgeEnabled() is False
        # SpaceSeparatedListOfStrings() returns a list-like; coerce for comparison.
        assert list(conf.supybot.plugins.LLM.bridgeAllowedPlugins()) == []

    def test_status_page_urls_default_is_three_urls(self) -> None:
        """A string default would become a list of single characters:
        registry.Value.__init__ passes the default to setValue, and
        SeparatedListOf.setValue calls list(v)."""
        import llm.config  # noqa: F401 — import side effect registers the values
        import supybot.conf as conf

        assert list(conf.supybot.plugins.LLM.statusPageUrls()) == [
            "Claude=https://status.claude.com",
            "GitHub=https://www.githubstatus.com",
            "OpenAI=https://status.openai.com",
        ]

    def test_status_queryable_pages_defaults_empty(self) -> None:
        import llm.config  # noqa: F401
        import supybot.conf as conf

        assert list(conf.supybot.plugins.LLM.statusQueryablePages()) == []

    def test_status_page_url_singular_is_gone(self) -> None:
        import llm.config  # noqa: F401
        import supybot.conf as conf

        assert not hasattr(conf.supybot.plugins.LLM, "statusPageUrl")

    def test_capability_settings_registered_with_defaults(self) -> None:
        """T5b: capability-based registry values are the sole surface for
        assistant/image model after the command-era keys were removed."""
        import llm.config  # noqa: F401 — import side effect registers values
        import supybot.conf as conf

        assert conf.supybot.plugins.LLM.assistantModel() == "anthropic/claude-haiku-5-5"
        assert conf.supybot.plugins.LLM.assistantSystemPrompt() != ""
        assert conf.supybot.plugins.LLM.imageModel() == "xai/grok-imagine-image"
        assert conf.supybot.plugins.LLM.refusalFallbackModel() == "xai/grok-4-1-fast-non-reasoning"

    def test_bridge_allow_mutating_registered_with_safe_default(self) -> None:
        """B1: bridgeAllowMutating defaults to False (gate closed).

        Note on the assertion shape: ``bridgeAllowMutating`` is a
        ``registerChannelValue`` (per-channel), but calling it as
        ``conf.supybot.plugins.LLM.bridgeAllowMutating()`` with no channel
        argument returns the channel-independent default. That is exactly
        what we want to verify here — that the registered default is
        ``False``, not the value for any particular channel.
        """
        import llm.config  # noqa: F401 — import side effect registers the value
        import supybot.conf as conf

        assert conf.supybot.plugins.LLM.bridgeAllowMutating() is False

    def test_bridge_scheduled_task_limit_default_is_five(self) -> None:
        """D1: bridgeScheduledTaskLimit defaults to 5 (per-creator-per-channel)."""
        import llm.config  # noqa: F401 — import side effect registers the value
        import supybot.conf as conf

        assert conf.supybot.plugins.LLM.bridgeScheduledTaskLimit() == 5

    def test_verse_enabled_registered_default_false(self) -> None:
        """C1: verseEnabled is a per-channel boolean that defaults to False."""
        import llm.config  # noqa: F401 — import side effect registers the value
        import supybot.conf as conf

        assert conf.supybot.plugins.LLM.verseEnabled() is False

    def test_verse_event_retention_days_registered_default_30(self) -> None:
        """C1: verseEventRetentionDays is a per-channel integer that defaults to 30."""
        import llm.config  # noqa: F401 — import side effect registers the value
        import supybot.conf as conf

        assert conf.supybot.plugins.LLM.verseEventRetentionDays() == 30


class TestRateLimitDefaults:
    """Pin the 18 rate-limit (key, default) pairs produced by _register_rate_limit_block."""

    @pytest.mark.parametrize(
        "key,expected",
        [
            ("askRateLimitCount", 15),
            ("askRateLimitWindow", 60),
            ("askTrustedRateLimitCount", 15),
            ("askTrustedRateLimitWindow", 60),
            ("askUnregRateLimitCount", 15),
            ("askUnregRateLimitWindow", 60),
            ("codeRateLimitCount", 10),
            ("codeRateLimitWindow", 60),
            ("codeTrustedRateLimitCount", 0),
            ("codeTrustedRateLimitWindow", 60),
            ("codeUnregRateLimitCount", 2),
            ("codeUnregRateLimitWindow", 60),
            ("drawRateLimitCount", 2),
            ("drawRateLimitWindow", 300),
            ("drawTrustedRateLimitCount", 5),
            ("drawTrustedRateLimitWindow", 60),
            # Unregistered is the STRICTEST tier: 1 image/hour, NOT 0 (which
            # _is_rate_limited treats as "disabled → unlimited").
            ("drawUnregRateLimitCount", 1),
            ("drawUnregRateLimitWindow", 3600),
            ("storyRateLimitCount", 2),
            ("storyRateLimitWindow", 300),
            ("storyTrustedRateLimitCount", 5),
            ("storyTrustedRateLimitWindow", 60),
            ("storyUnregRateLimitCount", 1),
            ("storyUnregRateLimitWindow", 3600),
        ],
    )
    def test_rate_limit_defaults(self, key: str, expected: int) -> None:
        """GIVEN rate-limit registry defaults WHEN read THEN match expected values."""
        import llm.config  # noqa: F401 — import side effect registers the values
        import supybot.conf as conf

        value = getattr(conf.supybot.plugins.LLM, key)()
        assert value == expected


class TestValidatedModelNameThreadSafety:
    """Test thread safety of ValidatedModelName._warned set."""

    def setup_method(self) -> None:
        """Clear warned-models cache between tests."""
        ValidatedModelName._warned.clear()

    def test_concurrent_set_value_no_duplicate_warnings(self, mocker: MockerFixture) -> None:
        """GIVEN many threads setting the same unpriced model WHEN concurrent THEN warns at most once."""
        warnings_logged: list[str] = []
        lock = threading.Lock()

        def capture_warning(msg: str, *args: object) -> None:
            with lock:
                warnings_logged.append(msg % args if args else msg)

        model = "xai/thread-safety-test-model"

        def set_model() -> None:
            v = ValidatedModelName("", "test")
            v.setValue(model)

        mocker.patch.object(
            logging.getLogger("supybot.plugins.LLM.config"),
            "warning",
            side_effect=capture_warning,
        )
        threads = [threading.Thread(target=set_model) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should warn exactly once despite 20 concurrent threads
        assert len(warnings_logged) == 1
        assert model in warnings_logged[0]


class TestPR3RegistryKeys:
    """F1: compaction registry keys."""

    def test_verse_compaction_daily_at_default(self) -> None:
        """verseCompactionDailyAt is global String, default '03:00'."""
        import llm.config  # noqa: F401
        import supybot.conf as conf

        assert conf.supybot.plugins.LLM.verseCompactionDailyAt() == "03:00"

    def test_verse_compaction_min_keep_events_default(self) -> None:
        """verseCompactionMinKeepEvents is global NonNegativeInteger, default 20."""
        import llm.config  # noqa: F401
        import supybot.conf as conf

        assert conf.supybot.plugins.LLM.verseCompactionMinKeepEvents() == 20


class TestVerseAutoEntityKeys:
    """Two new per-channel registry keys for verse_record + aging."""

    def test_verse_auto_entity_retire_days_default_14(self) -> None:
        """verseAutoEntityRetireDays defaults to 14 (per-channel)."""
        import llm.config  # noqa: F401
        import supybot.conf as conf

        assert conf.supybot.plugins.LLM.verseAutoEntityRetireDays() == 14

    def test_verse_auto_entity_max_names_per_call_default_8(self) -> None:
        """verseAutoEntityMaxNamesPerCall defaults to 8 (per-channel)."""
        import llm.config  # noqa: F401
        import supybot.conf as conf

        assert conf.supybot.plugins.LLM.verseAutoEntityMaxNamesPerCall() == 8

    def test_verse_auto_entity_retire_days_zero_allowed(self) -> None:
        """0 must be a valid value — design §5: '0 disables sweep'."""
        import llm.config  # noqa: F401
        import supybot.conf as conf

        conf.supybot.plugins.LLM.verseAutoEntityRetireDays.setValue(0)
        assert conf.supybot.plugins.LLM.verseAutoEntityRetireDays() == 0


class TestVerseStyleExemplars:
    """registry.Json channel key for fc42 taste exemplars."""

    def test_verse_style_exemplars_default_empty_list(self) -> None:
        """registry.Json round-trips the empty-list default cleanly."""
        from supybot import registry

        v = registry.Json([], "h")
        assert v() == []

    def test_verse_style_exemplars_json_roundtrips_quote_laden(self) -> None:
        """Strings with quotes and dashes survive the bot.conf read path."""
        from supybot import registry

        v = registry.Json([], "h")
        payload = ['"the lads marched," he said', "it's grim up north", "BRAAAP—ven"]
        v.setValue(payload)
        restored = registry.Json([], "h")
        restored.set(str(v))  # str(v) == json.dumps(...) == the bot.conf read path
        assert restored() == payload

    def test_verse_style_exemplars_registered_default_empty_list(self) -> None:
        """verseStyleExemplars per-channel default is []."""
        import llm.config  # noqa: F401
        import supybot.conf as conf

        assert conf.supybot.plugins.LLM.verseStyleExemplars() == []


def _registered_config_keys() -> set[str]:
    """Every key name registered against the LLM plugin in config.py."""
    path = LLM_SRC / "config.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    keys: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in REGISTER_METHODS
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            keys.add(node.args[1].value)
    return keys


def _registry_value_literals() -> list[tuple[Path, int, str]]:
    """Every string-literal key passed to a ``registryValue(...)`` call
    under plugin.py / service.py.

    A call that builds its key dynamically (an f-string, a bare variable, a
    ``PROFILES[...].overlay_setting`` lookup) is invisible to this scan and
    must be reviewed by hand -- this only guards literals, of which there are
    63 today against zero dynamic-key blind spots left unreviewed.
    """
    found: list[tuple[Path, int, str]] = []
    for fname in ("plugin.py", "service.py"):
        path = LLM_SRC / fname
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "registryValue"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                found.append((path, node.lineno, node.args[0].value))
    return found


class TestEveryRegistryValueLiteralIsRegistered:
    """The production incident this pins: a commit read
    ``registryValue("statusPageUrl")`` (singular) after config.py had
    already been renamed to ``statusPageUrls`` (plural). Every test in the
    suite passed --  ``make_registry_side_effect``'s stub just returns ""
    for a key it doesn't recognise -- while production raised
    ``NonExistentRegistryEntry`` on every chat message, because the real
    supybot registry has no fallback.

    This statically cross-checks every literal key plugin.py/service.py
    reads via ``registryValue`` against every key config.py actually
    registers, so a deleted or mistyped literal fails here instead of live.
    """

    def test_the_scan_actually_finds_calls(self) -> None:
        """Guards the guard: a broken scanner would make the real test
        below vacuously pass."""
        assert len(_registry_value_literals()) > 50
        assert len(_registered_config_keys()) > 50

    def test_no_orphaned_registry_value_literal(self) -> None:
        registered = _registered_config_keys()
        offenders = [
            f"{path.relative_to(REPO_ROOT)}:{lineno}: registryValue({key!r})"
            for path, lineno, key in _registry_value_literals()
            if key not in registered
        ]
        assert not offenders, (
            "registryValue() reads a key config.py never registers -- this "
            "raises NonExistentRegistryEntry in production, not in this "
            "test suite:\n" + "\n".join(offenders)
        )
