"""Tests for provider-scoped API key resolution."""

from __future__ import annotations

import io
import logging
import os
import sys
import threading
import types
from collections.abc import Generator

import pytest
from llm import apikeys


class TestProviderOf:
    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("xai/grok-4.3", "xai"),
            ("anthropic/claude-3-opus", "anthropic"),
            ("xai/grok-imagine-image", "xai"),
        ],
    )
    def test_known_models_resolve(self, model: str, expected: str) -> None:
        """GIVEN a provider/name model we call WHEN provider_of THEN its prefix."""
        assert apikeys.provider_of(model) == expected

    @pytest.mark.parametrize(
        "model",
        [
            "",
            "   ",
            "not-a-real-model-xyz",
            "claude-3-opus",
            "gpt-4",  # unprefixed: no provider we call
            "gemini/gemini-3-flash-preview",  # no longer a managed provider
            "openai/gpt-5.2",
            "vertex_ai/imagen-4.0-generate-001",
            "xai/",
            "/",
            "//",
            "/gpt-4",
            "a/b/c/d",
            "gpt-4\n",
            "model with spaces",
            "a" * 500,
        ],
    )
    def test_unresolvable_models_return_empty(self, model: str) -> None:
        """GIVEN a model naming no provider we call WHEN provider_of THEN "" and no raise.

        Key resolution runs inside failure handlers, so an exception here would
        surface as an unhandled crash instead of a config error.
        """
        assert apikeys.provider_of(model) == ""

    def test_mixed_case_prefix_resolves(self) -> None:
        """GIVEN "XAI/grok-4.3" WHEN provider_of THEN "xai".

        provider.chat lowercases the prefix too, so the key check and the call
        agree on which provider is being paid.
        """
        assert apikeys.provider_of("XAI/grok-4.3") == "xai"


class TestApiKeyFor:
    def test_returns_env_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN XAI_API_KEY set WHEN api_key_for an xai model THEN that value."""
        monkeypatch.setenv("XAI_API_KEY", "xai-fake-value-for-tests")
        assert apikeys.api_key_for("xai/grok-4.3") == "xai-fake-value-for-tests"

    def test_missing_var_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN XAI_API_KEY unset WHEN api_key_for THEN None."""
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        assert apikeys.api_key_for("xai/grok-4.3") is None

    @pytest.mark.parametrize("value", ["", "   "])
    def test_blank_var_returns_none(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        """GIVEN a blank var WHEN api_key_for THEN None, not "".

        "" would read as configured at the guard and then fail at the provider.
        """
        monkeypatch.setenv("XAI_API_KEY", value)
        assert apikeys.api_key_for("xai/grok-4.3") is None

    def test_value_is_stripped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN a trailing newline WHEN api_key_for THEN stripped.

        docker --env-file and hand-edited files both produce these.
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "  sk-ant-fake-value-for-tests\n")
        assert apikeys.api_key_for("anthropic/claude-3-opus") == "sk-ant-fake-value-for-tests"

    def test_unmanaged_provider_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN an unmanaged model WHEN api_key_for THEN None even if a var exists.

        Guards against an implementation that derives the variable name as
        f"{provider.upper()}_API_KEY" instead of consulting the map.
        """
        monkeypatch.setenv("VERTEX_AI_API_KEY", "should-not-be-used-anywhere")
        assert apikeys.api_key_for("vertex_ai/imagen-4.0-generate-001") is None

    def test_provider_isolation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN both vars set WHEN resolving each THEN no cross-provider bleed."""
        monkeypatch.setenv("XAI_API_KEY", "xai-fake-value-for-tests")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-fake-value-for-tests")
        assert apikeys.api_key_for("xai/grok-4.3") == "xai-fake-value-for-tests"
        assert apikeys.api_key_for("anthropic/claude-3-opus") == "anthropic-fake-value-for-tests"

    def test_reads_env_on_every_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN the var changes WHEN resolving again THEN the new value.

        Kills a memoized implementation: redaction reads the same source, so a
        cached key could be sent while an uncached one is scrubbed.
        """
        monkeypatch.setenv("XAI_API_KEY", "first-fake-value-here")
        assert apikeys.api_key_for("xai/grok-4.3") == "first-fake-value-here"
        monkeypatch.setenv("XAI_API_KEY", "second-fake-value-here")
        assert apikeys.api_key_for("xai/grok-4.3") == "second-fake-value-here"


class TestKnownSecretValues:
    def test_collects_provider_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN provider vars set WHEN collected THEN both included."""
        monkeypatch.setenv("XAI_API_KEY", "xai-fake-value-long-enough")
        monkeypatch.setenv("GEMINI_API_KEY", "gemini-fake-value-long-enough")
        values = apikeys.known_secret_values()
        assert "xai-fake-value-long-enough" in values, "provider value missing (set withheld)"
        assert "gemini-fake-value-long-enough" in values, "provider value missing (set withheld)"

    @pytest.mark.parametrize(
        "name", ["SOME_API_KEY", "HF_TOKEN", "CLIENT_SECRET", "GOOGLE_APPLICATION_CREDENTIALS"]
    )
    def test_collects_by_suffix(self, monkeypatch: pytest.MonkeyPatch, name: str) -> None:
        """GIVEN a var with a secret suffix WHEN collected THEN included."""
        monkeypatch.setenv(name, "some-credential-value-long-enough")
        assert "some-credential-value-long-enough" in apikeys.known_secret_values(), (
            "suffixed value missing (set withheld)"
        )

    def test_ignores_unrelated_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN a non-secret var WHEN collected THEN excluded."""
        monkeypatch.setenv("EDITOR", "a-value-that-is-long-enough")
        assert "a-value-that-is-long-enough" not in apikeys.known_secret_values(), (
            "unrelated value included (set withheld)"
        )

    @pytest.mark.parametrize(("length", "included"), [(15, False), (16, True)])
    def test_length_floor_boundary(
        self, monkeypatch: pytest.MonkeyPatch, length: int, included: bool
    ) -> None:
        """GIVEN a value at the boundary WHEN collected THEN >= floor is included.

        Pins the comparison: 8, >, and >= all pass a test that only uses 8 and 26.
        """
        monkeypatch.setenv("FOO_API_KEY", "x" * length)
        assert (("x" * length) in apikeys.known_secret_values()) is included

    def test_ignores_blank_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN an empty secret var WHEN collected THEN no empty string.

        An empty entry would make scrub() insert [REDACTED] between every char.
        """
        monkeypatch.setenv("BAR_API_KEY", "")
        assert "" not in apikeys.known_secret_values()

    def test_matches_conftest_suffixes(self) -> None:
        """GIVEN the conftest isolation fixture WHEN comparing THEN suffixes agree.

        conftest duplicates this tuple (it must not import apikeys). If they
        drift, the fixture stops scrubbing something redaction thinks it covers.
        """
        from .conftest import _SECRET_SUFFIXES

        assert set(_SECRET_SUFFIXES) == set(apikeys.SECRET_SUFFIXES)


class TestSecretVarNames:
    def test_returns_names_not_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN a secret var WHEN listing names THEN the name, never the value.

        This is logged at startup. A mutant returning values turns the
        mitigation into the leak.
        """
        monkeypatch.setenv("XAI_API_KEY", "xai-fake-value-long-enough")
        names = apikeys.secret_var_names()
        assert "XAI_API_KEY" in names
        assert not any("xai-fake-value-long-enough" in name for name in names)


class TestScrub:
    def test_replaces_every_occurrence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN the key twice WHEN scrub THEN both replaced.

        Kills a .replace(secret, REDACTED, 1) implementation.
        """
        monkeypatch.setenv("XAI_API_KEY", "xai-fake-value-long-enough")
        text = "key xai-fake-value-long-enough and again xai-fake-value-long-enough"
        result = apikeys.scrub(text)
        assert "xai-fake-value-long-enough" not in result
        assert result.count("[REDACTED]") == 2

    def test_handles_none(self) -> None:
        """GIVEN None WHEN scrub THEN empty string."""
        assert apikeys.scrub(None) == ""

    def test_passthrough_without_secrets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN no secret vars WHEN scrub THEN text unchanged."""
        for name in list(apikeys.PROVIDER_ENV_VARS.values()):
            monkeypatch.delenv(name, raising=False)
        assert apikeys.scrub("nothing to hide here") == "nothing to hide here"


class TestSecretFilter:
    SECRET = "xai-fake-value-long-enough"

    @staticmethod
    def _record(msg: object, args: object = None, exc_info: object = None) -> logging.LogRecord:
        # Construct with args=None and set record.args directly afterward.
        # LogRecord.__init__ has a special case for a *single Mapping passed
        # as the sole positional arg* (logger.error("%(k)s", {"k": v})): it
        # unwraps args=({"k": v},) into record.args = {"k": v}. That unwrap
        # indexes args[0], so handing it a bare dict/Mapping of length 1
        # directly (rather than tuple-wrapped) raises KeyError before the
        # record even exists. Setting record.args post-construction bypasses
        # that unwrap and lets every test pass record.args in the exact final
        # shape it wants to exercise (dict, MappingProxyType, tuple, None).
        record = logging.LogRecord(
            name="LLM.test",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg=msg,
            args=None,
            exc_info=exc_info,
        )
        record.args = args
        return record

    @pytest.fixture(autouse=True)
    def _set_secret(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XAI_API_KEY", self.SECRET)

    def test_scrubs_message(self) -> None:
        """GIVEN a key in record.msg WHEN filtered THEN redacted."""
        record = self._record(f"call failed with {self.SECRET}")
        assert apikeys.SecretFilter().filter(record) is True
        assert self.SECRET not in record.getMessage()

    def test_scrubs_non_string_msg(self) -> None:
        """GIVEN an exception as record.msg WHEN filtered THEN redacted.

        log.error(exc) is legal and skips any isinstance(msg, str) check.
        """
        record = self._record(ValueError(f"bad key {self.SECRET}"))
        apikeys.SecretFilter().filter(record)
        assert self.SECRET not in record.getMessage()

    def test_scrubs_exception_object_args(self) -> None:
        """GIVEN an exception as an arg WHEN filtered THEN redacted.

        log.error("failed: %s", exc) is the dominant pattern in this codebase
        (service.py:5453,5696,5715; plugin.py:1023,1955) and provider
        AuthenticationError bodies echo the submitted key.
        """
        record = self._record("save failed: %s", (ValueError(f"bad key {self.SECRET}"),))
        apikeys.SecretFilter().filter(record)
        assert self.SECRET not in record.getMessage()

    def test_scrubs_string_args(self) -> None:
        """GIVEN a key in a string arg WHEN filtered THEN redacted."""
        record = self._record("%s", (f"api_key='{self.SECRET}'",))
        apikeys.SecretFilter().filter(record)
        assert self.SECRET not in record.getMessage()

    def test_dict_args_still_format(self) -> None:
        """GIVEN a dict arg WHEN filtered THEN scrubbed and still formattable."""
        record = self._record("%(k)s", {"k": f"key {self.SECRET}"})
        apikeys.SecretFilter().filter(record)
        assert self.SECRET not in record.getMessage()

    def test_mapping_args_do_not_break_formatting(self) -> None:
        """GIVEN a non-dict Mapping arg WHEN filtered THEN getMessage still works.

        logging unwraps a lone Mapping into record.args; treating it as a tuple
        iterates its KEYS and getMessage() then raises "format requires a
        mapping". Redaction must never break logging.
        """
        record = self._record("%(k)s", types.MappingProxyType({"k": f"key {self.SECRET}"}))
        apikeys.SecretFilter().filter(record)
        message = record.getMessage()
        assert self.SECRET not in message

    def test_non_string_args_survive_formatting(self) -> None:
        """GIVEN numeric args WHEN filtered THEN formatting is unaffected."""
        record = self._record("took %d ms", (42,))
        apikeys.SecretFilter().filter(record)
        assert record.getMessage() == "took 42 ms"

    def test_scrubs_traceback(self) -> None:
        """GIVEN a key in the exception WHEN filtered THEN exc_text redacted."""

        try:
            raise ValueError(f"auth failed for {self.SECRET}")
        except ValueError:
            record = self._record("boom", exc_info=sys.exc_info())
        apikeys.SecretFilter().filter(record)
        assert record.exc_text is not None
        assert self.SECRET not in record.exc_text

    def test_scrubs_stack_info(self) -> None:
        """GIVEN a key in stack_info WHEN filtered THEN redacted."""
        record = self._record("boom")
        record.stack_info = f"  line with {self.SECRET}"
        apikeys.SecretFilter().filter(record)
        assert self.SECRET not in record.stack_info

    def test_never_drops_records(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN no secrets configured WHEN filtered THEN record still emitted, unchanged.

        A deploy missing --env-file runs with no keys at all, which is exactly
        when its logs matter most. Every suffix-matching var is cleared, not
        just the provider ones: conftest also sets GEMINI/OPENAI fakes, and any
        one of them leaves redaction with something to do.
        """
        for name in list(os.environ):
            if name.upper().endswith(apikeys.SECRET_SUFFIXES):
                monkeypatch.delenv(name, raising=False)
        assert apikeys.known_secret_values() == set()
        record = self._record("took %d ms for %s", (42, "plain"))
        assert apikeys.SecretFilter().filter(record) is True
        assert record.getMessage() == "took 42 ms for plain"

    def test_survives_concurrent_env_mutation(self) -> None:
        """GIVEN env churn on another thread WHEN filtering THEN no RuntimeError.

        known_secret_values iterates os.environ, which wraps a plain dict;
        iterating during another thread's setenv raises "dictionary changed size
        during iteration". Production never mutates env, but the test suite does
        constantly, and threaded tests log.
        """
        stop = threading.Event()
        errors: list[BaseException] = []

        def churn() -> None:
            while not stop.is_set():
                os.environ["CHURN_API_KEY"] = "churn-value-long-enough"
                os.environ.pop("CHURN_API_KEY", None)

        def filt() -> None:
            try:
                for _ in range(2000):
                    apikeys.SecretFilter().filter(self._record("msg"))
            except BaseException as exc:  # noqa: BLE001 — recording it is the assertion
                errors.append(exc)

        churner = threading.Thread(target=churn, daemon=True)
        filterer = threading.Thread(target=filt)
        churner.start()
        filterer.start()
        filterer.join()
        stop.set()
        churner.join(timeout=2)
        assert not errors, f"filter raised under concurrent env mutation: {errors[:1]}"


class TestEndToEndRedaction:
    def test_supybot_exception_is_scrubbed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN a real logger.exception WHEN handled THEN no key in the output.

        The only test that exercises filter -> Filterer -> Formatter as
        production does. Every other filter test hand-builds a LogRecord and so
        cannot catch a filter that is never invoked, or a formatter that
        re-renders from exc_info.
        """
        secret = "xai-fake-value-long-enough"
        monkeypatch.setenv("XAI_API_KEY", secret)
        buffer = io.StringIO()
        handler = logging.StreamHandler(buffer)
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler.addFilter(apikeys.SecretFilter())
        logger = logging.getLogger("llm.test.e2e")
        logger.propagate = False
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        try:
            try:
                raise ValueError(f"401: bad key {secret}")
            except ValueError:
                logger.exception("call failed: %s", secret)
        finally:
            logger.removeHandler(handler)
        output = buffer.getvalue()
        assert secret not in output
        assert "[REDACTED]" in output


class TestInstallSecretFilter:
    """install_secret_filter is Task 4's job to call; this module only owns
    correctness of the function itself (target loggers, idempotency, count).

    Global logging-filter state (root/supybot/llm handlers, logging.lastResort)
    mutated by install_secret_filter() is snapshotted and restored by the
    suite-wide autouse fixture in the top-level conftest.py — this class no
    longer needs a class-local copy of that protection.
    """

    @pytest.fixture
    def target_handler(self) -> Generator[logging.Handler]:
        """A handler on one of install_secret_filter's real targets ("llm")."""
        logger = logging.getLogger("llm")
        handler = logging.StreamHandler(io.StringIO())
        logger.addHandler(handler)
        try:
            yield handler
        finally:
            logger.removeHandler(handler)

    @pytest.mark.parametrize(
        "logger_name",
        ["", "supybot", "llm"],
    )
    def test_installs_on_each_target_logger(self, logger_name: str) -> None:
        """GIVEN a handler on each real install target WHEN installed THEN it gets a SecretFilter.

        Parametrized over every target, not just one: a mutant that shrinks
        `targets` to `["llm"]` alone still passes if only "llm" is checked —
        and per the production leak this module exists to close, "supybot" is
        the target that actually has a handler attached today.
        """
        logger = logging.getLogger(logger_name)
        handler = logging.StreamHandler(io.StringIO())
        logger.addHandler(handler)
        try:
            installed = apikeys.install_secret_filter()
            assert installed >= 1
            assert any(isinstance(f, apikeys.SecretFilter) for f in handler.filters)
        finally:
            logger.removeHandler(handler)

    def test_installs_on_last_resort(self) -> None:
        """GIVEN no handler anywhere in the llm.* hierarchy WHEN installed
        THEN logging.lastResort still gets a SecretFilter.

        This is the actual fallback production hits today: nothing in this
        repo attaches a handler to root or to "llm", so an "llm.verse.*"
        record with no handler anywhere in its ancestry is handled by
        logging.lastResort — a bare stderr handler owned by no logger, which
        install_secret_filter must filter directly rather than assuming some
        logger's handler will always be there to catch it.
        """
        assert logging.lastResort is not None
        installed = apikeys.install_secret_filter()
        assert installed >= 1
        assert any(isinstance(f, apikeys.SecretFilter) for f in logging.lastResort.filters)

    def test_handles_missing_last_resort(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN logging.lastResort is None WHEN installed THEN no raise.

        Something else in the process (a test, a library) could set
        logging.lastResort = None, disabling Python's own fallback handler.
        install_secret_filter must not assume it's always present.
        """
        monkeypatch.setattr(logging, "lastResort", None)
        apikeys.install_secret_filter()

    def test_idempotent_does_not_double_install(self, target_handler: logging.Handler) -> None:
        """GIVEN it already ran WHEN run again THEN no duplicate filter, 0 newly installed.

        Startup can call this more than once (e.g. re-registration on
        @reload); a duplicate filter would just waste cycles, not leak
        anything, but it's still the wrong behaviour to pin.
        """
        first = apikeys.install_secret_filter()
        second = apikeys.install_secret_filter()
        assert first >= 1
        assert second == 0
        assert sum(isinstance(f, apikeys.SecretFilter) for f in target_handler.filters) == 1
