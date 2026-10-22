# Provider-scoped API keys implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace twelve per-channel, per-role API-key registry entries with one environment variable per provider, resolved from the model being called, at the four functions that actually talk to a provider.

**Architecture:** A new pure module `llm/apikeys.py` maps a model to its provider via `litellm.get_llm_provider()` and then to a single environment variable. The four outbound boundaries — `_timed_completion`, `_xai_responses_call`, `_generate_image_once`, `LiteLLMVerseClient.call` — resolve the key themselves, so no caller can pass a mismatched one. Providers outside the four mapped ones resolve to `None`, which lets LiteLLM use its own native credentials. A `logging.Filter` installed on handlers scrubs secrets from log output. The four `*ApiKey` registry settings are then deleted.

**Tech stack:** Python 3.12-3.14, Limnoria (supybot), LiteLLM 1.93.0 (locked), pytest, uv, ruff, ty.

**Spec:** `docs/superpowers/specs/2026-08-01-provider-scoped-api-keys-design.md`

## Global constraints

- Repo root: `/Users/rdrake/workspace/afternet/vibebot-v8`. Source: `plugins/llm/src/llm/`. Tests: `plugins/llm/tests/`.
- Run everything from the repo root — a hook runs `make lint && make typecheck` after every edit and fails from any other directory.
- `make test` runs `-m "not slow"`: 2572 collected, 2558 run, 14 deselected. **`test_stress.py` is entirely `slow`** and never runs there. To exercise it: `uv run pytest plugins/llm/tests/test_stress.py -m slow`.
- Coverage gate is `fail_under = 93`, currently 94%. Measured against 7920 statements, no step in this plan can breach it — if `make test` fails, it is a real test failure, not the gate.
- Final gate is `make preflight` (formats, then lint + format-check + typecheck + syntax-check + test), not just `make check`.
- Never log, print, or assert on a real API key. Tests use fakes only. This includes verification commands: compare a hash, never a prefix.
- `service.py` uses **relative** imports (`from .profile import ...`). Use `from .apikeys import ...`.
- `_` is `PluginInternationalization("LLM")` (`service.py:64`); `%`-style args are the file's convention.
- Environment variables, exact spelling: `XAI_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`.
- **Line numbers shift as you edit.** Every reference below was verified against the current tree, but re-grep between edits rather than trusting an offset.
- Commit after each task. Direct commits to `main`; no PR.

---

### Task 1: Test-environment isolation

**Files:**
- Modify: `plugins/llm/tests/conftest.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `FAKE_PROVIDER_KEYS: dict[str, str]` importable by other test modules; an autouse fixture guaranteeing a known environment.

**This is Task 1 for a reason.** `litellm/__init__.py:27` calls `load_dotenv()` at import, and `.env` is gitignored — so importing `llm.service` puts a developer's **real** keys into `os.environ` before collection. Any test that asserts on a collected secret set would render those keys into a pytest failure diff. The fixture must exist before such a test does.

Verified: no file under `plugins/llm/src/` or `plugins/llm/tests/` reads or writes `os.environ` today, so this changes no existing behaviour.

- [ ] **Step 1: Add the fixture**

In `plugins/llm/tests/conftest.py`, at module level (imports at the top of the file, not mid-module — ruff `E402` is enabled):

```python
import os

# Fake values, each comfortably over apikeys.MIN_REDACTABLE_LEN.
FAKE_PROVIDER_KEYS = {
    "XAI_API_KEY": "xai-fake-key-for-tests-0000",
    "GEMINI_API_KEY": "AIza-fake-key-for-tests-0000",
    "OPENAI_API_KEY": "sk-fake-key-for-tests-0000",
    "ANTHROPIC_API_KEY": "sk-ant-fake-key-for-tests-0000",
}

_SECRET_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET", "_CREDENTIALS")


@pytest.fixture(autouse=True)
def _isolate_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every test a known, fake set of provider credentials.

    LiteLLM calls load_dotenv() at import, so a developer's real .env is in
    os.environ before collection. Without this, tests can pass locally against
    real keys, fail in CI, print real keys into a failure diff, or reach the
    network.
    """
    for name in list(os.environ):
        if name.upper().endswith(_SECRET_SUFFIXES):
            monkeypatch.delenv(name, raising=False)
    for name, value in FAKE_PROVIDER_KEYS.items():
        monkeypatch.setenv(name, value)
```

`_SECRET_SUFFIXES` is duplicated here rather than imported from `llm.apikeys` because that module does not exist until Task 2 and this fixture must not depend on it. Task 2 adds a test asserting the two stay in sync.

- [ ] **Step 2: Add a network guard**

The suite has no socket blocking (no `pytest-socket`, no `responses`, no VCR). Combined with Step 1 handing every test plausible-looking credentials, any incompletely-mocked path becomes a live outbound request. Add to the same file:

```python
@pytest.fixture(autouse=True)
def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly instead of making a real request from a test."""
    import socket

    def _refuse(*args: object, **kwargs: object) -> None:
        raise RuntimeError(
            "test attempted a real network connection — mock the provider call"
        )

    monkeypatch.setattr(socket.socket, "connect", _refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", _refuse)
```

- [ ] **Step 3: Run the full suite**

```bash
make test
```

Expected: 2558 passed. Any failure is a fixture bug or a test that was quietly reaching the network — investigate rather than weakening the fixture. If a test legitimately needs a socket (none are known), give it an opt-out marker rather than removing the guard.

- [ ] **Step 4: Commit**

```bash
git add plugins/llm/tests/conftest.py
git commit -m "test: isolate provider credentials and block network from tests"
```

---

### Task 2: `apikeys.py` — model-to-key resolution

**Files:**
- Create: `plugins/llm/src/llm/apikeys.py`
- Test: `plugins/llm/tests/test_apikeys.py`

**Interfaces:**
- Consumes: `FAKE_PROVIDER_KEYS` (Task 1).
- Produces: `PROVIDER_ENV_VARS: dict[str, str]`, `provider_of(model) -> str`, `env_var_for(model) -> str | None`, `api_key_for(model) -> str | None`, `is_managed(model) -> bool`.

- [ ] **Step 1: Write the failing test**

Create `plugins/llm/tests/test_apikeys.py`:

```python
"""Tests for provider-scoped API key resolution."""

from __future__ import annotations

import pytest

from llm import apikeys


class TestProviderOf:
    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("xai/grok-4.3", "xai"),
            ("gemini/gemini-3-flash-preview", "gemini"),
            ("openai/gpt-5.2", "openai"),
            ("anthropic/claude-3-opus", "anthropic"),
            ("gpt-4", "openai"),          # unprefixed names are legal
            ("dall-e-3", "openai"),
            ("vertex_ai/imagen-4.0-generate-001", "vertex_ai"),  # known, unmanaged
        ],
    )
    def test_known_models_resolve(self, model: str, expected: str) -> None:
        """GIVEN a model LiteLLM recognises WHEN provider_of THEN its provider.

        vertex_ai is included deliberately: "known but unmanaged" and
        "unresolvable" must stay distinguishable, because they take different
        paths at the boundary.
        """
        assert apikeys.provider_of(model) == expected

    @pytest.mark.parametrize(
        "model",
        [
            "", "   ", "not-a-real-model-xyz", "claude-3-opus",
            "/", "//", "/gpt-4", "a/b/c/d", "gpt-4\n", "model with spaces", "a" * 500,
        ],
    )
    def test_unresolvable_models_return_empty(self, model: str) -> None:
        """GIVEN a model LiteLLM rejects WHEN provider_of THEN "" and no raise.

        LiteLLM raises BadRequestError for these. Key resolution runs inside
        failure handlers, so a new exception type here would surface as an
        unhandled crash instead of a config error.
        """
        assert apikeys.provider_of(model) == ""

    def test_mixed_case_prefix_is_not_resolvable(self) -> None:
        """GIVEN "XAI/grok-4.3" WHEN provider_of THEN "" — LiteLLM is case-sensitive.

        Pinned so the behaviour is deliberate: normalizing the case here would
        only send LiteLLM a model string it rejects a moment later.
        """
        assert apikeys.provider_of("XAI/grok-4.3") == ""


class TestEnvVarFor:
    def test_managed_provider(self) -> None:
        """GIVEN an xai model WHEN env_var_for THEN XAI_API_KEY."""
        assert apikeys.env_var_for("xai/grok-4.3") == "XAI_API_KEY"

    def test_unmanaged_provider(self) -> None:
        """GIVEN a vertex_ai model WHEN env_var_for THEN None."""
        assert apikeys.env_var_for("vertex_ai/imagen-4.0-generate-001") is None


class TestIsManaged:
    def test_managed(self) -> None:
        """GIVEN an xai model WHEN is_managed THEN True."""
        assert apikeys.is_managed("xai/grok-4.3") is True

    @pytest.mark.parametrize(
        "model", ["vertex_ai/imagen-4.0-generate-001", "not-a-real-model-xyz", ""]
    )
    def test_unmanaged(self, model: str) -> None:
        """GIVEN an unmanaged or unresolvable model WHEN is_managed THEN False.

        False means "hand LiteLLM None and let it use its own credentials" —
        which is how vertex_ai keeps working via ADC.
        """
        assert apikeys.is_managed(model) is False


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
        monkeypatch.setenv("GEMINI_API_KEY", "  AIza-fake-value-for-tests\n")
        assert apikeys.api_key_for("gemini/gemini-3-flash-preview") == "AIza-fake-value-for-tests"

    def test_unmanaged_provider_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN a vertex_ai model WHEN api_key_for THEN None even if a var exists.

        Guards against an implementation that derives the variable name as
        f"{provider.upper()}_API_KEY" instead of consulting the map.
        """
        monkeypatch.setenv("VERTEX_AI_API_KEY", "should-not-be-used-anywhere")
        assert apikeys.api_key_for("vertex_ai/imagen-4.0-generate-001") is None

    def test_provider_isolation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN both vars set WHEN resolving each THEN no cross-provider bleed."""
        monkeypatch.setenv("XAI_API_KEY", "xai-fake-value-for-tests")
        monkeypatch.setenv("GEMINI_API_KEY", "gemini-fake-value-for-tests")
        assert apikeys.api_key_for("xai/grok-4.3") == "xai-fake-value-for-tests"
        assert apikeys.api_key_for("gemini/gemini-3-flash-preview") == "gemini-fake-value-for-tests"

    def test_reads_env_on_every_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN the var changes WHEN resolving again THEN the new value.

        Kills a memoized implementation: redaction reads the same source, so a
        cached key could be sent while an uncached one is scrubbed.
        """
        monkeypatch.setenv("XAI_API_KEY", "first-fake-value-here")
        assert apikeys.api_key_for("xai/grok-4.3") == "first-fake-value-here"
        monkeypatch.setenv("XAI_API_KEY", "second-fake-value-here")
        assert apikeys.api_key_for("xai/grok-4.3") == "second-fake-value-here"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest plugins/llm/tests/test_apikeys.py -v
```

Expected: `ModuleNotFoundError: No module named 'llm.apikeys'`.

- [ ] **Step 3: Write the implementation**

Create `plugins/llm/src/llm/apikeys.py`:

```python
"""Provider-scoped API key resolution.

A key is a property of which provider is being paid, not of which channel or
which role asked. This module maps a model identifier to its provider and then
to a single environment variable, so the key is always anchored to the model
actually being called.

Pure and dependency-light on purpose: no plugin, no service, no Limnoria
registry. That keeps it unit-testable and keeps key resolution out of
``service.py``.
"""

from __future__ import annotations

import os

import litellm

# The providers this deployment pays for directly. Anything else LiteLLM
# recognises — vertex_ai, openrouter, azure, bedrock — resolves to None so
# LiteLLM uses its own native credential mechanism (ADC, IAM, its own env vars).
# Narrowing that to an allowlist would turn a multi-provider plugin into a
# four-provider one.
PROVIDER_ENV_VARS: dict[str, str] = {
    "xai": "XAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def provider_of(model: str) -> str:
    """Return the LiteLLM provider for ``model``, or "" if unresolvable.

    Uses ``litellm.get_llm_provider`` rather than splitting on "/" because
    unprefixed model names are legal — LiteLLM resolves ``gpt-4`` and
    ``dall-e-3`` to openai — and this plugin's config validator accepts anything
    LiteLLM accepts.

    Never raises. LiteLLM raises ``BadRequestError`` for names it cannot place,
    and callers sit inside failure handlers where a new exception type would
    surface as an unhandled crash rather than a configuration error.
    """
    if not model or not model.strip():
        return ""
    try:
        return str(litellm.get_llm_provider(model)[1])
    except Exception:  # noqa: BLE001 — an unplaceable model is a config error, not a crash
        return ""


def env_var_for(model: str) -> str | None:
    """Environment variable name ``model`` needs, or None if unmanaged.

    Returns the *name*, never the value, so callers can tell an operator which
    variable to set.
    """
    return PROVIDER_ENV_VARS.get(provider_of(model))


def is_managed(model: str) -> bool:
    """True if this deployment supplies ``model``'s credential directly.

    False means "pass None and let LiteLLM resolve" — the path that keeps
    vertex_ai (ADC), openrouter, azure and bedrock working.
    """
    return provider_of(model) in PROVIDER_ENV_VARS


def api_key_for(model: str) -> str | None:
    """Configured API key for ``model``'s provider, or None.

    Read from the environment on every call rather than cached, so the value
    redaction scrubs can never diverge from the value actually sent.
    """
    name = env_var_for(model)
    if not name:
        return None
    return os.environ.get(name, "").strip() or None
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest plugins/llm/tests/test_apikeys.py -v
```

Expected: all PASS. If `anthropic/claude-3-opus` fails, the installed LiteLLM does not know that name — substitute one it does know rather than changing `provider_of`.

- [ ] **Step 5: Lint, typecheck, commit**

```bash
make lint && make typecheck
git add plugins/llm/src/llm/apikeys.py plugins/llm/tests/test_apikeys.py
git commit -m "feat(apikeys): resolve API keys from the model's provider"
```

---

### Task 3: `apikeys.py` — secret discovery and `SecretFilter`

**Files:**
- Modify: `plugins/llm/src/llm/apikeys.py`
- Test: `plugins/llm/tests/test_apikeys.py`

**Interfaces:**
- Consumes: `PROVIDER_ENV_VARS`.
- Produces: `SECRET_SUFFIXES`, `MIN_REDACTABLE_LEN`, `known_secret_values() -> set[str]`, `secret_var_names() -> list[str]`, `scrub(text: str | None) -> str`, `SecretFilter`, `install_secret_filter() -> int`.

**Put `import logging` in the module header**, not mid-file — ruff `E402` is enabled and the post-edit hook runs `make lint`.

- [ ] **Step 1: Write the failing test**

Append to `plugins/llm/tests/test_apikeys.py`:

```python
import io
import logging
import threading
import types


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
        from conftest import _SECRET_SUFFIXES

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
        return logging.LogRecord(
            name="LLM.test", level=logging.ERROR, pathname=__file__, lineno=1,
            msg=msg, args=args, exc_info=exc_info,
        )

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
        import sys

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
        """GIVEN no secrets configured WHEN filtered THEN record still emitted."""
        for name in list(apikeys.PROVIDER_ENV_VARS.values()):
            monkeypatch.delenv(name, raising=False)
        assert apikeys.SecretFilter().filter(self._record("plain")) is True

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
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest plugins/llm/tests/test_apikeys.py -k "Secret or Scrub or EndToEnd" -v
```

Expected: `AttributeError: module 'llm.apikeys' has no attribute 'known_secret_values'`.

- [ ] **Step 3: Write the implementation**

Add `import logging` and `from collections.abc import Mapping` to the module header of `apikeys.py`, then append:

```python
# Credentials do not all end in _API_KEY. A provider whose secret is named
# differently would otherwise sit outside redaction entirely.
SECRET_SUFFIXES: tuple[str, ...] = ("_API_KEY", "_TOKEN", "_SECRET", "_CREDENTIALS")

# Real keys are far longer (Gemini 39, xAI ~84). The floor stops a short junk
# value such as FOO_API_KEY=disabled turning redaction into find-and-replace on
# a common word across every log line.
MIN_REDACTABLE_LEN = 16

REDACTED = "[REDACTED]"


def _secret_items() -> list[tuple[str, str]]:
    """(name, stripped value) for every environment secret worth redacting."""
    items = []
    for name, raw in list(os.environ.items()):  # list() — env can mutate concurrently
        if not name.upper().endswith(SECRET_SUFFIXES):
            continue
        value = raw.strip()
        if len(value) >= MIN_REDACTABLE_LEN:
            items.append((name, value))
    return items


def known_secret_values() -> set[str]:
    """Every environment value that must never appear in output."""
    return {value for _name, value in _secret_items()}


def secret_var_names() -> list[str]:
    """Sorted names of the variables redaction covers. Names only, never values."""
    return sorted(name for name, _value in _secret_items())


def scrub(text: str | None) -> str:
    """Replace every known secret value in ``text`` with ``[REDACTED]``."""
    if not text:
        return ""
    result = str(text)
    for secret in known_secret_values():
        result = result.replace(secret, REDACTED)
    return result


class SecretFilter(logging.Filter):
    """Strip API keys from log records before they are formatted.

    Covers message, arguments, traceback and stack info, because supybot's
    ``Logger.exception`` uses all of them: it writes the raw traceback and calls
    ``collect_extra_debug_data()``, a repr of every frame local and every
    attribute of ``self``.

    Arguments are scrubbed by their ``str()`` value rather than by type: an
    exception object carries the provider's error body, which routinely echoes
    the submitted key.

    Never drops a record, and never breaks formatting — redaction must not cost
    observability. Value replacement is defence in depth, not a boundary: it
    cannot catch an encoded or truncated credential.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not known_secret_values():
            return True
        record.msg = scrub(str(record.msg)) if record.msg is not None else record.msg
        record.args = self._scrub_args(record.args)
        if record.exc_info and not record.exc_text:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = scrub(record.exc_text)
        if record.stack_info:
            record.stack_info = scrub(record.stack_info)
        return True

    @staticmethod
    def _scrub_args(args: object) -> object:
        """Scrub arguments without changing their shape.

        A lone Mapping must stay a Mapping: logging unwraps it into ``args``,
        and turning it into a tuple of keys makes ``getMessage()`` raise
        "format requires a mapping".
        """
        if not args:
            return args
        if isinstance(args, Mapping):
            return {key: scrub(str(value)) for key, value in args.items()}
        return tuple(scrub(str(arg)) if not isinstance(arg, (int, float)) else arg for arg in args)


def install_secret_filter() -> int:
    """Attach ``SecretFilter`` to every output handler, idempotently.

    Handlers, not loggers: a logger's filters run only for records that
    originate on it, and propagation to an ancestor runs the ancestor's
    *handlers*, not its *filters*. This plugin logs through at least ten loggers
    across two hierarchies (``supybot.plugins.LLM.*`` and ``llm.verse.*``), so
    per-logger installation would cover whichever two we happened to name.

    Handlers created later — supybot adds per-plugin file handlers when
    ``individualLogfiles`` is true; prod has it false — are not covered. Calling
    this again picks them up.

    Returns the number of handlers newly filtered, for the startup log line.
    """
    installed = 0
    targets = [logging.getLogger(), logging.getLogger("supybot"), logging.getLogger("llm")]
    for logger in targets:
        for handler in logger.handlers:
            if not any(isinstance(existing, SecretFilter) for existing in handler.filters):
                handler.addFilter(SecretFilter())
                installed += 1
    return installed
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest plugins/llm/tests/test_apikeys.py -v
```

Expected: all PASS. `test_non_string_args_survive_formatting` requires the `(int, float)` passthrough — without it `%d` receives a string and formatting raises.

- [ ] **Step 5: Lint, typecheck, commit**

```bash
make lint && make typecheck
git add plugins/llm/src/llm/apikeys.py plugins/llm/tests/test_apikeys.py
git commit -m "feat(apikeys): scrub secrets from log records at the handler layer"
```

---

### Task 4: Redaction cutover

**Files:**
- Modify: `plugins/llm/src/llm/service.py:1186-1191` (`_API_KEY_NAMES`), `:1193-1220` (`_configured_api_keys`), `:1222-1240` (`_sanitize`), `plugins/llm/src/llm/plugin.py` (`__init__`, near `:730`)
- Test: `plugins/llm/tests/test_service_core.py:218-300`

**Interfaces:**
- Consumes: `apikeys.scrub`, `apikeys.install_secret_filter`, `apikeys.secret_var_names`.
- Produces: `_sanitize` unchanged in signature, backed by the environment.

**Must land before Task 8.** Deleting the registry settings first would leave `_configured_api_keys` walking settings that no longer exist — and both its registry reads are wrapped in bare `except Exception`, so it would return an empty set and degrade `_sanitize` to a silent pass-through.

- [ ] **Step 1: Update the tests**

There are **six** `_sanitize` tests at `test_service_core.py:218,233,248,253,270,289`, in a class that uses `self.service` / `self.mock_plugin` from a setup fixture — they do not call `make_service()`. Match that style.

Delete `test_api_key_sanitization_channel_specific_key` (`:289`): it calls
`conf.supybot.plugins.LLM.get("assistantApiKey").get("#forest").setValue(...)`, which raises `NonExistentRegistryEntry` after Task 8, and per-channel keys no longer exist as a concept. Replace with:

```python
    def test_sanitize_sources_from_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GIVEN a key only in the environment WHEN sanitizing THEN redacted.

        Asserts both directions in one test: unset first (not redacted), then
        set (redacted). Either half alone can pass for the wrong reason.
        """
        text = "AuthenticationError: key xai-fake-key-for-tests-0000 rejected"
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        assert "xai-fake-key-for-tests-0000" in self.service._sanitize(text)
        monkeypatch.setenv("XAI_API_KEY", "xai-fake-key-for-tests-0000")
        assert "xai-fake-key-for-tests-0000" not in self.service._sanitize(text)
        assert "[REDACTED]" in self.service._sanitize(text)
```

Also repoint `test_provider_edge_cases.py:506` (`test_api_key_sanitized_in_errors`), which injects `fake_key = "sk-" + "x" * 25` through `make_service(assistantApiKey=fake_key)` — a registry-sourced key is no longer redactable. Set it via `monkeypatch.setenv("OPENAI_API_KEY", fake_key)` instead (`TEST_MODEL` is `gpt-4` → openai).

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest plugins/llm/tests/test_service_core.py -k sanitize -v
```

Expected: the new environment test FAILS (registry-sourced redaction does not see the env value).

- [ ] **Step 3: Implement**

- Delete `_API_KEY_NAMES` (`1186-1191`) and `_configured_api_keys` (`1193-1220`).
- Rewrite `_sanitize` (`1222-1240`) as `return apikeys.scrub(text)`, keeping the docstring's explanation of why value replacement beats regex, and adding that the environment is now the source.
- In `plugin.py.__init__`, after `self.log.addFilter(TraceFilter())`, install the filter and log its coverage — names only:

```python
        installed = apikeys.install_secret_filter()
        self.log.info(
            "secret redaction: %d handler(s) filtered, %d variable(s) covered: %s",
            installed,
            len(apikeys.secret_var_names()),
            ", ".join(apikeys.secret_var_names()) or "none",
        )
```

Note for the reviewer: `install_secret_filter` attaches to handlers on the root, `supybot`, and `llm` loggers. Do **not** replace this with per-logger `addFilter` calls — `limnoria_bridge.py:23` is a module-level logger with no `__init__` to hook, and the five `llm.verse.*` loggers are in a different hierarchy from `supybot.*`.

- [ ] **Step 4: Run the tests**

```bash
uv run pytest plugins/llm/tests/test_service_core.py plugins/llm/tests/test_provider_edge_cases.py -v && make test
```

Expected: PASS.

- [ ] **Step 5: Lint, typecheck, commit**

```bash
make lint && make typecheck
git add plugins/llm/src/llm/service.py plugins/llm/src/llm/plugin.py plugins/llm/tests/
git commit -m "feat(logging): redact secrets from env at the handler layer

Hand-applied _sanitize never covered supybot's Logger.exception, which writes
the raw traceback plus a repr of every frame local."
```

---

### Task 5: Resolve keys at the outbound boundaries

**Files:**
- Modify: `plugins/llm/src/llm/service.py` — `_timed_completion` (`2351`, litellm call at `2378`), `_xai_responses_call` (`3281`, call at `3322`), `_generate_image_once` (`4150`, call at `4172`), plus the guard sites `2637`, `2718`, `3028`, `3602`, `3787`, `4090`, `4325`, `5117` and the key reads at `3182-3184`, `3672`, `6124`, `6195`
- Test: `test_provider_edge_cases.py`, `test_service_completion.py`, `test_service_images.py`, `test_service_memory.py`, `test_reminders.py`, `test_etiquette.py`, `test_stress.py`

**Interfaces:**
- Consumes: `apikeys.api_key_for`, `apikeys.is_managed`, `apikeys.env_var_for`.
- Produces: `LLMService._missing_key_error(self, model: str) -> str | None` — the operator-facing message, or `None` when the model needs no managed key.

**This is the breaking commit.** It cannot be split: no test-only commit can express "the key comes from the model" while the code still reads the registry, so any split guarantees a red intermediate. It absorbs the stress-test rewrite for the same reason.

- [ ] **Step 1: Write the failing tests**

Add to `plugins/llm/tests/test_provider_edge_cases.py`. **`make_service` is a fixture (`conftest.py:508`), not a module function** — every test must request it:

```python
class TestBoundaryKeyResolution:
    """The key litellm receives is the one the model's provider variable holds."""

    def test_xai_model_gets_xai_key(self, make_service, monkeypatch) -> None:
        """GIVEN an xai model WHEN completing THEN XAI_API_KEY reaches litellm."""
        monkeypatch.setenv("XAI_API_KEY", "xai-fake-key-for-tests-0000")
        service, _ = make_service(assistantModel="xai/grok-4.3")
        seen: dict[str, object] = {}
        monkeypatch.setattr(
            "litellm.completion",
            lambda **kwargs: seen.update(kwargs) or _stub_response(),
        )
        service._timed_completion("ask", model="xai/grok-4.3", messages=[], channel=None)
        assert seen["api_key"] == "xai-fake-key-for-tests-0000"

    def test_gemini_model_gets_gemini_key(self, make_service, monkeypatch) -> None:
        """GIVEN a gemini model WHEN completing THEN the gemini key, not the xai one."""
        monkeypatch.setenv("XAI_API_KEY", "xai-fake-key-for-tests-0000")
        monkeypatch.setenv("GEMINI_API_KEY", "AIza-fake-key-for-tests-0000")
        service, _ = make_service()
        seen: dict[str, object] = {}
        monkeypatch.setattr(
            "litellm.completion",
            lambda **kwargs: seen.update(kwargs) or _stub_response(),
        )
        service._timed_completion(
            "ask", model="gemini/gemini-3-flash-preview", messages=[], channel=None
        )
        assert seen["api_key"] == "AIza-fake-key-for-tests-0000"

    def test_unmanaged_provider_passes_none(self, make_service, monkeypatch) -> None:
        """GIVEN a vertex_ai model WHEN completing THEN api_key is None.

        None is what makes LiteLLM fall back to ADC. Sending "" or a mapped
        key here would break service-account auth.
        """
        service, _ = make_service()
        seen: dict[str, object] = {}
        monkeypatch.setattr(
            "litellm.completion",
            lambda **kwargs: seen.update(kwargs) or _stub_response(),
        )
        service._timed_completion(
            "ask", model="vertex_ai/gemini-2.0-flash", messages=[], channel=None
        )
        assert seen["api_key"] is None

    def test_missing_key_error_names_provider_and_variable(self, make_service, monkeypatch) -> None:
        """GIVEN no key WHEN building the error THEN it names both."""
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        service, _ = make_service()
        message = service._missing_key_error("xai/grok-4.3")
        assert message is not None
        assert "xai" in message
        assert "XAI_API_KEY" in message

    @pytest.mark.parametrize("model", ["vertex_ai/imagen-4.0-generate-001", "", "junk-model"])
    def test_unmanaged_models_have_no_error(self, make_service, model) -> None:
        """GIVEN an unmanaged model WHEN building the error THEN None.

        Unmanaged is not a failure — it is delegation to LiteLLM.
        """
        service, _ = make_service()
        assert service._missing_key_error(model) is None

    def test_error_never_contains_a_key(self, make_service, monkeypatch) -> None:
        """GIVEN a configured key WHEN building any error THEN the value is absent."""
        monkeypatch.setenv("XAI_API_KEY", "xai-fake-key-for-tests-0000")
        service, _ = make_service()
        assert "xai-fake-key-for-tests-0000" not in (service._missing_key_error("xai/grok-4.3") or "")
```

Write `_stub_response()` as a small local helper returning an object shaped like a litellm completion response — copy the existing mock shape from `test_service_completion.py` rather than inventing one. Adjust `_timed_completion`'s keyword names to its real signature at `service.py:2351`; read it first.

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest plugins/llm/tests/test_provider_edge_cases.py -k BoundaryKey -v
```

Expected: `AttributeError: ... has no attribute '_missing_key_error'`.

- [ ] **Step 3: Add the helper**

In `service.py`, where `_API_KEY_NAMES` used to be:

```python
    def _missing_key_error(self, model: str) -> str | None:
        """Message for a managed provider whose variable is unset, else None.

        None means the provider is not one we supply credentials for, so the key
        is LiteLLM's problem (ADC, IAM, its own environment variables) and there
        is nothing to report.
        """
        if not apikeys.is_managed(model) or apikeys.api_key_for(model):
            return None
        return _("no API key configured for provider '%s' (set %s)") % (
            apikeys.provider_of(model),
            apikeys.env_var_for(model),
        )
```

Add `from .apikeys import ...` (relative — house style) or `from . import apikeys`.

- [ ] **Step 4: Resolve at the three service boundaries**

In each, resolve from the model the function is about to send and drop any `api_key` parameter:

- `_timed_completion` (`2351`): set `kwargs["api_key"] = apikeys.api_key_for(model)` before `litellm.completion` at `2378`.
- `_xai_responses_call` (`3281`): same, before `litellm.responses` at `3322`. Delete its `api_key` parameter (`3286`).
- `_generate_image_once` (`4150`): replace `api_key=self.plugin.registryValue("imageApiKey", channel)` at `4175` with `api_key=apikeys.api_key_for(model)`.

Then delete the now-redundant key threading: `_completion_with_tool_fallback`'s `api_key` parameter (`1866`, used at `1903`/`1921`), and the key reads at `3182-3184`, `3672`, `6124`, `6195`.

At `3182-3184` only those three lines change — `3179-3181` already computes the model correctly. The collapse is a bug fix: the two chains resolved independently, so grounded search on a grok channel sent a Gemini key to an xAI model.

- [ ] **Step 5: Convert the eight guards**

Each keeps its existing return shape and message channel; only the condition and text change:

```python
key_error = self._missing_key_error(model)
if key_error:
    # existing failure return, with reason=/content= set to _("Error: %s") % key_error
```

| Site | Model to resolve from |
|---|---|
| `2637` | `task.model` |
| `2718` | `task.model` |
| `3028` | **`model_override or registryValue(model_name, channel)`** — hoist `3036` above the guard. The registry value alone is wrong whenever an override is in play. |
| `3602` | `registryValue("assistantModel", target)` — hoist above the guard |
| `3787` | `registryValue("assistantModel", target)` — hoist above the guard |
| `4090` | `registryValue("assistantModel", target)` — hoist above the guard |
| `4325` | `model` from `4324` (already `model_override or ...`) |
| `5117` | `registryValue("imageModel", channel)` — hoist above the guard |

For `2637`/`2718`, `task.model` is a **persisted DB column**. A row queued before the deploy with an empty model resolves to unmanaged → no error → the call proceeds and LiteLLM reports the real problem. That is the correct outcome; do not add a hard failure there. `pending_tasks` is the live `@ask`/`@code`/`@draw` timeout-recovery queue, so a hard failure would silently kill in-flight user requests across the deploy.

- [ ] **Step 6: Migrate the affected tests**

Guards that **stop** firing — convert each from an empty registry stub to `monkeypatch.delenv(<provider var>)`, where the variable matches the test's model:

- `test_provider_edge_cases.py:491,499` (`assistantApiKey=None`/`""`)
- `test_service_images.py:971` (`test_auto_rewrite_skipped_when_ask_key_missing`), `:1145`, `:1318`
- `test_service_completion.py:1587` (`{"codeApiKey": ""}`)
- `test_service_memory.py:49,63`

Exact-kwarg assertions — **the expected variable differs per test**, check each test's model:

- `test_service_completion.py:117` → model is `xai/grok-4.3` → `FAKE_PROVIDER_KEYS["XAI_API_KEY"]`
- `test_reminders.py:721` → model is `gemini/gemini-flash-latest` → `FAKE_PROVIDER_KEYS["GEMINI_API_KEY"]`
- `test_service_memory.py:105,725,757,806` → `TEST_MODEL` is `gpt-4` → `FAKE_PROVIDER_KEYS["OPENAI_API_KEY"]`

Assert on `==` with the expected value, never `is not None`.

Positive stubs that only need their assertion values repointed (not a `delenv`): `test_service_memory.py:24,113,148`. Note `:24` is inside an autouse `setup(self, make_service, mocker)` fixture that takes no `monkeypatch`.

`conftest.py:474` (`"searchApiKey": ""`) — remove; the chain it fed is gone.

`test_service_images.py:800-825` (`vertex_ai/...`, 14 tests in `TestDrawAutoRewrite`) and `test_etiquette.py:119` (bare `"imagen"`) need **no change** under the delegate-the-rest rule — they resolve to unmanaged, pass `None`, and their mocks are unaffected. Run them and confirm rather than editing.

- [ ] **Step 7: Rewrite the stress isolation test**

`test_stress.py:264-314` gives 20 threads 20 distinct `assistantApiKey` values and asserts `len(unique_keys) == 20`. Replace it with a version that still drives `service.completion()` against a mocked `litellm.completion` and observes the kwarg actually forwarded — asserting on `_api_key_for` directly would test a pure function and prove nothing:

```python
    def test_completion_key_isolation_across_providers(self, make_service, monkeypatch) -> None:
        """GIVEN concurrent requests on different providers THEN no key bleeds."""
        models = ["xai/grok-4.3", "gemini/gemini-3-flash-preview", "openai/gpt-5.2"]
        expected = {
            "xai/grok-4.3": FAKE_PROVIDER_KEYS["XAI_API_KEY"],
            "gemini/gemini-3-flash-preview": FAKE_PROVIDER_KEYS["GEMINI_API_KEY"],
            "openai/gpt-5.2": FAKE_PROVIDER_KEYS["OPENAI_API_KEY"],
        }
        observed: list[tuple[str, object]] = []
        lock = threading.Lock()

        def record(**kwargs: object) -> object:
            with lock:
                observed.append((kwargs["model"], kwargs.get("api_key")))
            return _stub_response()

        monkeypatch.setattr("litellm.completion", record)
        service, _ = make_service()

        def worker(index: int) -> None:
            model = models[index % len(models)]
            service._timed_completion("ask", model=model, messages=[], channel=None)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(observed) == 20
        assert len({key for _model, key in observed}) == 3
        for model, key in observed:
            assert key == expected[model], f"key bleed: {model} received another provider's key"
```

`test_stress.py` builds `LLMService(mock_plugin)` by hand and has neither `make_service` nor `FAKE_PROVIDER_KEYS` in scope — import what you need. `threading` is already imported (`:9`).

- [ ] **Step 8: Run everything, including the slow tests**

```bash
make test
uv run pytest plugins/llm/tests/test_stress.py -m slow -v
```

Both must pass. `make test` alone does **not** cover `test_stress.py`.

- [ ] **Step 9: Lint, typecheck, commit**

```bash
make lint && make typecheck
git add plugins/llm/src/llm/service.py plugins/llm/tests/
git commit -m "refactor(service): resolve API keys at the outbound boundaries

Anchoring the key to the model at the point of the call fixes two live
mismatches — the pending-task retry paired codeApiKey with assistantModel, and
grounded search resolved its key and its model independently — and makes a
caller-side mismatch unrepresentable."
```

---

### Task 6: Resolve the verse compaction key at its boundary

**Files:**
- Modify: `plugins/llm/src/llm/verse/compaction.py:273,276,278-287`, `plugins/llm/src/llm/plugin.py:6347-6349`
- Test: `plugins/llm/tests/test_plugin_verse.py`

**Interfaces:**
- Consumes: `apikeys.api_key_for`.
- Produces: `LiteLLMVerseClient()` taking no `api_key`.

- [ ] **Step 1: Write the failing test**

`compaction.py` imports litellm **inside** `call()` (`:281`) — there is no module-level `compaction.litellm`, so patching that attribute raises `AttributeError`, and a half-patched test would make a **real HTTPS call to Google**. Patch the litellm module itself:

```python
class TestCompactionKeyResolution:
    def test_key_matches_the_call_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN a gemini compaction model WHEN calling THEN the gemini key is sent."""
        from llm.verse import compaction

        monkeypatch.setenv("GEMINI_API_KEY", "AIza-fake-key-for-tests-0000")
        monkeypatch.setenv("XAI_API_KEY", "xai-fake-key-for-tests-0000")
        seen: dict[str, object] = {}

        def fake_completion(**kwargs: object) -> object:
            seen.update(kwargs)
            raise RuntimeError("stop here — only the kwargs matter")

        monkeypatch.setattr("litellm.completion", fake_completion)
        client = compaction.LiteLLMVerseClient()
        with pytest.raises(RuntimeError):
            client.call(op="compact", model="gemini/gemini-flash-lite-latest", messages=[])
        assert seen["api_key"] == "AIza-fake-key-for-tests-0000"

    def test_client_takes_no_api_key_argument(self) -> None:
        """GIVEN the constructor WHEN inspected THEN no api_key parameter."""
        import inspect

        from llm.verse import compaction

        assert "api_key" not in inspect.signature(compaction.LiteLLMVerseClient).parameters
```

Read the real `def call()` at `compaction.py:278-280` first and match its keywords.

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest plugins/llm/tests/test_plugin_verse.py -k CompactionKeyResolution -v
```

- [ ] **Step 3: Implement**

- `compaction.py`: delete the `api_key` parameter (`:273`) and `self._api_key = api_key or None` (`:276`). Keep `def call()` (`278-280`). Replace the conditional `kwargs["api_key"] = self._api_key` (`285-286`) with `kwargs["api_key"] = apikeys.api_key_for(model)`.
- `plugin.py:6348`: delete the `api_key = self.registryValue("assistantApiKey")` line; construct `LiteLLMVerseClient()` with no arguments at `:6349`.

Storing the key on `self` also fed supybot's attribute walk on exceptions — removing it is a redaction improvement as well as a correctness one.

- [ ] **Step 4: Run the tests**

```bash
uv run pytest plugins/llm/tests/test_plugin_verse.py -v && make test
```

- [ ] **Step 5: Lint, typecheck, commit**

```bash
make lint && make typecheck
git add plugins/llm/src/llm/verse/compaction.py plugins/llm/src/llm/plugin.py plugins/llm/tests/test_plugin_verse.py
git commit -m "fix(verse): resolve the compaction key from the compaction model

The client paired a gemini/ model with assistantApiKey, an xAI key in prod."
```

---

### Task 7: Delete `profile.api_key_setting`

**Files:**
- Modify: `plugins/llm/src/llm/profile.py:60,80,93,101-102,106,118,132,157`
- Test: `plugins/llm/tests/test_profile.py:62,127,182`, `test_assistant.py:4378,4399,4439,4441,4451,4515,4596,4658`

- [ ] **Step 1: Update the tests**

- `test_profile.py` — delete `test_api_key_setting_is_a_known_registry_key` (`:62`) and `test_api_key_setting_value` (`:182`); they are two differently-named parametrized methods, 10 cases total. Delete the `EXPECTED_API_KEY` table (`:127`).
- `test_assistant.py:4378` — delete the two key assertions at `:4439` and `:4441`; keep the model assertions, which are the point of the test.
- `test_assistant.py:4399,4451,4515,4596,4658` — remove `api_key_setting=` from five `Profile(...)` constructions. Four are overlay/model tests, pure collateral.

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest plugins/llm/tests/test_profile.py plugins/llm/tests/test_assistant.py -v
```

Expected: FAIL — `Profile.__init__` still requires the field.

- [ ] **Step 3: Implement**

Delete the `api_key_setting: str` field (`:80`), its docstring line (`:60`), the five `api_key_setting="assistantApiKey",` entries, and the stale comment at `:101-102` naming `assistantApiKey`/`codeApiKey`.

```bash
grep -rn api_key_setting plugins/llm/ | grep -v __pycache__
```

Expected: no hits.

- [ ] **Step 4: Run the full suite**

```bash
make test
```

- [ ] **Step 5: Lint, typecheck, commit**

```bash
make lint && make typecheck
git add plugins/llm/src/llm/profile.py plugins/llm/tests/
git commit -m "refactor(profile): drop api_key_setting, identical on every profile"
```

---

### Task 8: Delete the registry key settings

**Files:**
- Modify: `plugins/llm/src/llm/config.py:120,131-171`
- Test: `plugins/llm/tests/test_config.py:137,159,196-209`, `conftest.py:403,406,417`, `test_plugin_verse.py:2806,2855`

- [ ] **Step 1: Confirm nothing reads them**

```bash
grep -rn "ApiKey" plugins/llm/src/ | grep -v __pycache__
```

Expected: only `service.py:2995`, a docstring reading ``instead of the registry ``{command}ApiKey`` value``. Delete that line too — Task 5 removed the parameter it documents. Any other hit means a Task 5 site was missed; fix it there.

- [ ] **Step 2: Update the tests**

- `conftest.py:403,406,417` — delete the three remaining key stubs (`:474` went in Task 5).
- `test_config.py:137` — wizard string assertion; update to the new text.
- `test_config.py:159` — `codeApiKey._private`; delete.
- `test_config.py:196-209` — **delete the whole `test_capability_api_keys_are_private` method (202-209)**, not just its two asserts; removing only the asserts leaves an empty body and a `SyntaxError`. In the defaults test, delete the four key entries.
- `test_plugin_verse.py:2806,2855` — delete the `if key == "assistantApiKey"` branches from the bare-LLM stubs.

- [ ] **Step 3: Implement**

In `config.py`, delete the four `registerChannelValue` blocks (`131-171`) and the `# API Keys` header above them. Replace the wizard's key line (`:120`):

```python
    print("\nAPI keys come from the environment, one per provider:")
    print("  XAI_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY")
```

**Do not change the `imageModel` default.** `vertex_ai/imagen-4.0-generate-001` resolves to unmanaged, passes `None`, and LiteLLM uses ADC — it works, and prod carries its own explicit value regardless.

- [ ] **Step 4: Run the full suite**

```bash
make test
```

`conftest.py:496` returns `""` for unknown registry keys rather than raising, so most tests are unaffected; only those touching the real registry break.

- [ ] **Step 5: Lint, typecheck, commit**

```bash
make lint && make typecheck
git add plugins/llm/src/llm/config.py plugins/llm/tests/
git commit -m "feat(config)!: delete the four API-key registry settings

Keys come from one environment variable per provider, which makes per-channel
key drift structurally impossible."
```

---

### Task 9: Delete the dead key-plumbing parameters

**Files:**
- Modify: `plugins/llm/src/llm/service.py:2969,3028,3500,3544,4242,4325,2631-2636,3021,3025,4150,4158`

**Verified dead:** no caller in `plugins/llm` passes `api_key` to these three; the `**` spreads at their call sites expand `_pending_task_fns` (callables only); no `functools.partial` or dynamic dispatch reaches them.

- [ ] **Step 1: Re-verify**

```bash
grep -rn "api_key=" plugins/llm/src plugins/llm/tests | grep -v __pycache__
```

Expected: only the boundary assignments from Tasks 5 and 6, and test assertions. If a caller passes `api_key=` to `assistant_completion`, `assistant_request`, or `completion`, stop — the premise is wrong.

- [ ] **Step 2: Delete**

- `2969`/`3028` — `assistant_completion`'s `api_key` parameter and its `or` line.
- `3500`/`3544` — the facade's parameter and the forward.
- `4242`/`4325` — the profile path's parameter and its `or` line.
- `2631-2636` — the whole `api_key_name` ladder.
- `3021`, `3025` — the two `api_key_name = ...` lines **only**. The surrounding `if/else` also selects `codeModel`/`codeSystemPrompt`; deleting the block breaks the code path.
- `4150`/`4158` — `_generate_image_once`'s `channel` parameter and its docstring line. It existed solely for the per-channel `imageApiKey` lookup and passed a raw `msg.args[0]` — a nick in a PM — as a registry scope, unlike every other site. Update its callers.

- [ ] **Step 3: Run, lint, typecheck, commit**

```bash
make test && make lint && make typecheck
git add plugins/llm/src/llm/service.py plugins/llm/tests/
git commit -m "refactor(service): delete key-plumbing that reconciled per-role keys"
```

---

### Task 10: Documentation and operator runbook

**Files:**
- Modify: `.env.example`, `README.md:34,55`, `docs/guide/operator/configuration.md:25-28,33`, `docs/guide/operator/tuning-monitoring.md:22,25,166`, `docs/guide/operator/memory-promotion.md:88`, `docs/guide/operator/operations.md`

- [ ] **Step 1: Rewrite `.env.example`**

It says "these env vars are optional — API keys can be configured via bot.conf instead", which becomes false. `Makefile:243-249` copies this file to the live env path, so it is not merely illustrative. State: one variable per provider, required for the four managed providers, no bot.conf alternative; other providers use their own LiteLLM-native credentials. Include the `docker run --env-file` format rules — no quotes, no trailing comments, no spaces around `=`, LF endings — because that parser is stricter than a shell and a malformed file crashloops the container.

- [ ] **Step 2: Update the guides**

- `configuration.md:25-28,33` — replace the key table and `@config` example with the environment variables.
- `tuning-monitoring.md:22,25` — remove the `searchApiKey` → `assistantApiKey` fallback prose.
- `tuning-monitoring.md:166` — **the one that matters most**: it tells an operator to run `@config plugins.LLM.assistantApiKey` to confirm a key is set, which is exactly the diagnostic someone reaches for during this migration and exactly what stops working. Replace with a container-side check that never prints key material:

```
docker exec vibebot python3 -c "import hashlib,os;k=os.environ.get('XAI_API_KEY','');print(len(k), hashlib.sha256(k.encode()).hexdigest()[:12])"
```

- `memory-promotion.md:88` — drop the key reference.
- `README.md:34` — replace the `@config` key command. `README.md:55` — remove the claim that keys live in the registry.

- [ ] **Step 3: Add a rollback section and the exposure note to `operations.md`**

There is no rollback section today. A revert push is 10-20 minutes and can be blocked by one flaky test, since `docker.yml` gates on CI across a three-version matrix. Document the image pin (every build is tagged `type=sha`, `docker.yml:41-46`):

```bash
# Stop the updater first: vibebot-updater.service hardcodes :latest and would
# otherwise bounce the pinned container every 15 minutes.
systemctl --user stop vibebot-updater.timer
mkdir -p ~/.config/systemd/user/vibebot.service.d
printf '[Service]\nEnvironment=IMAGE=ghcr.io/rdrake/vibebot:sha-<PREV>\n' \
  > ~/.config/systemd/user/vibebot.service.d/override.conf
systemctl --user daemon-reload && systemctl --user restart vibebot
```

Document finding `<PREV>`: `docker inspect vibebot --format '{{.Config.Image}}'` before deploying, or the Actions run summary.

Add a short "credentials in the environment" note: `docker inspect` now returns them under `.Config.Env` (anyone in the `docker` group, and routinely pasted into tickets), `/proc/<pid>/environ` exposes them on the host, and child processes inherit them.

- [ ] **Step 4: Build the docs and commit**

```bash
make docs
git add .env.example README.md docs/guide/operator/
git commit -m "docs: describe provider environment variables and add a rollback runbook"
```

---

### Task 11: Final verification

- [ ] **Step 1: Full gate**

```bash
make preflight
```

`preflight` formats first, then runs lint, format-check, typecheck, syntax-check and the suite. Expected: 2558 passed, coverage at or above 93%.

- [ ] **Step 2: Slow tests**

```bash
uv run pytest plugins/llm/tests/test_stress.py -m slow -v
```

`make test` excludes these. Expected: PASS, including the rewritten isolation test.

- [ ] **Step 3: Confirm the deletions**

```bash
grep -rn "ApiKey\|api_key_setting" plugins/llm/src/ docs/ README.md .env.example | grep -v __pycache__
```

Expected: no hits outside `CHANGELOG.md` (historical) and the spec/plan documents.

- [ ] **Step 4: Confirm coverage of the new module**

```bash
uv run pytest plugins/llm/tests/test_apikeys.py --cov=llm.apikeys --cov-report=term-missing
```

Expected: at or near 100%. `apikeys.py` is pure; anything uncovered is a missing test.

- [ ] **Step 5: Prove redaction end to end**

```bash
uv run pytest plugins/llm/tests/ -k "secret or sanitize or redact or EndToEnd" -v
```

Expected: PASS, including the real-`logger.exception` case.

- [ ] **Step 6: Push**

```bash
git status --short
git push origin main
```

Wait for **both** CI and the Docker build — separate workflows, and auto-deploy fires on the Docker build's success, not CI's.

---

## Migration checklist (operator, after the code is deployed)

Not part of the code change. Preconditions first; step 1 is the highest outage risk in the exercise.

**Preconditions**

- [ ] **Set `supybot.commands.allowShell: False`** (currently `True`, `bot.conf:153`), with the bot stopped. `Debug`'s `environ` command replies `repr(os.environ)` **to the channel** and is one `@load` away for an owner. Before this change the worst owner fat-finger was one key delivered to a PM; afterwards it is all four keys in a channel. Treat as a precondition, not a follow-up.
- [ ] **Check `supybot.flush` and `supybot.upkeepInterval`** are not overridden in prod `bot.conf`. If `flush` is false, the automatic cleanup below does not happen and the manual sweep becomes mandatory.
- [ ] **Back up:** copy `bot.conf` off the host; record KEY-A through KEY-D in a password manager; record the deployed image tag (`docker inspect vibebot --format '{{.Config.Image}}'`). Limnoria's rolling backups roll forward and will all be post-deletion within a couple of flush cycles.

**Steps**

- [ ] **Populate the env file** (`/home/vibebot/.config/vibebot/env`): `GEMINI_API_KEY` = KEY-A, `XAI_API_KEY` = **KEY-B**. Give `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` real values or delete them. No quotes, no trailing comments, no spaces around `=`, LF endings, no stray `IMAGE=` line. `EnvironmentFile=-` tolerates a broken file; `docker run --env-file` does not, and `Restart=always` with no healthcheck turns a malformed file into a silent crashloop. Restart, then verify **from inside the container** without printing key material:
  `docker exec vibebot python3 -c "import hashlib,os;k=os.environ.get('XAI_API_KEY','');print(len(k), hashlib.sha256(k.encode()).hexdigest()[:12])"`
- [ ] **Deploy the code.** Confirm the startup line reporting how many handlers are filtered and how many variables redaction covers.
- [ ] **Verify within the first hour**, before the upkeep flush rewrites `bot.conf`: `@ask` on a grok channel, `@ask` on the global Gemini path, `@draw`, one grounded-search request.
- [ ] **Sweep `bot.conf` by prefix, bot stopped** — every `supybot.plugins.LLM.*ApiKey*` line, including the stale `askApiKey`/`drawApiKey` entries from settings renamed long ago. Expect the flush to have removed most already.
- [ ] **Revoke** KEY-C and KEY-D, plus whatever the stale `askApiKey`/`drawApiKey` entries held.
